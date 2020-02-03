from collections import deque

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import resize_boxes

from tracktor.training_set_generation import replicate_and_randomize_boxes
from tracktor.utils import clip_boxes
from tracktor.visualization import plot_bounding_boxes, VisdomLinePlotter
from tracktor.live_dataset import IndividualDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps, im_info,
                 transformed_image_size, box_roi_pool=None):
        self.id = track_id
        self.pos = pos
        self.score = score
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        self.last_pos = deque([pos.clone()], maxlen=mm_steps + 1)
        self.frames_since_active = 1
        self.last_v = torch.Tensor([])
        self.transformed_image_size = transformed_image_size
        self.gt_id = None
        self.im_info = im_info
        self.box_predictor_classification = None
        self.box_head_classification = None
        self.box_predictor_regression = None
        self.box_head_regression = None
        self.scale = self.im_info[0] / self.transformed_image_size[0][0]
       # self.plotter = VisdomLinePlotter(env_name='validation_over_time')
        self.checkpoints = dict()
        self.training_set_classification = IndividualDataset(self.id)
        self.training_set_regression = IndividualDataset(self.id)
        self.box_roi_pool = box_roi_pool
        self.use_for_finetuning = False
        self.ground_truth_box = None

    def has_positive_area(self):
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

    def add_features(self, features):
        """Adds new appearance features to the object."""
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object"""
        if len(self.features) > 1:
            features = torch.cat(list(self.features), dim=0)
        else:
            features = self.features[0]
        features = features.mean(0, keepdim=True)
        dist = F.pairwise_distance(features, test_features, keepdim=True)
        return dist

    def reset_last_pos(self):
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())

    # TODO is displacement of roi helpful? Kinda like dropout as specific features might not be in the ROI anymore
    # TODO only take negative examples that are close to positive example --> Makes training easier.
    # TODO try lower learning rate and not to overfit --> best behaviour of 6 was when 0 track still had high score.
    def generate_training_set_regression(self, gt_pos, max_displacement, batch_size=8, plot=False, plot_args=None):
        gt_pos = gt_pos.to(device)
        random_displaced_bboxes = replicate_and_randomize_boxes(gt_pos,
                                                                batch_size=batch_size,
                                                                max_displacement=max_displacement).to(device)

        training_boxes = clip_boxes(random_displaced_bboxes, self.im_info)

        if plot and plot_args:
            plot_bounding_boxes(self.im_info, gt_pos, plot_args[0], training_boxes.numpy(), plot_args[1], plot_args[2])

        return training_boxes

    def update_training_set_regression(self, ground_truth_box, batch_size, max_displacement, fpn_features,
                                           include_previous_frames=False, shuffle=True):
        boxes = self.generate_training_set_regression(ground_truth_box, max_displacement, batch_size, fpn_features)
        if shuffle:
            boxes = boxes[torch.randperm(boxes.size(0))]
        boxes_resized = resize_boxes(boxes, self.im_info, self.transformed_image_size[0])
        proposals = [boxes_resized]
        with torch.no_grad():
            roi_pool_feat = self.box_roi_pool(fpn_features, proposals, self.im_info).to(device)
        ground_truth_boxes = ground_truth_box.repeat(boxes.size()[0], 1)
        training_set_dict = {'features': roi_pool_feat, 'boxes': boxes,
                             'scores': torch.ones(boxes.size()[0]).to(device), 'ground_truth_boxes': ground_truth_boxes}

        if not include_previous_frames:
            self.training_set_regression = IndividualDataset(self.id)
        self.training_set_regression.append_samples(training_set_dict)

    def generate_training_set_classification(self, batch_size, additional_dets, fpn_features, shuffle=False):
        num_positive_examples = int(batch_size / 2)
        positive_examples = self.generate_training_set_regression(self.pos,
                                                                  0.0,
                                                                  batch_size=num_positive_examples).to(device)
        positive_examples = clip_boxes(positive_examples, self.im_info)
        # positive_examples = self.pos.repeat(num_positive_examples, 1)
        positive_examples = torch.cat((positive_examples, torch.ones([num_positive_examples, 1]).to(device)), dim=1)
        boxes = positive_examples
        if additional_dets.size(0) != 0:
            standard_batch_size_negative_example = int(np.floor(num_positive_examples / len(additional_dets)))
            offset = num_positive_examples - (standard_batch_size_negative_example * additional_dets.size(0))
            for i in range(additional_dets.size(0)):
                num_negative_example = standard_batch_size_negative_example
                if offset != 0:
                    num_negative_example += 1
                    offset -= 1
                if num_negative_example == 0:
                    break
                negative_example = self.generate_training_set_regression(additional_dets[i].view(1, -1),
                                                                         0.0,
                                                                         batch_size=num_negative_example).to(device)
                negative_example = clip_boxes(negative_example, self.im_info)
                # negative_example = additional_dets[i].view(1, -1).repeat(num_negative_example, 1)
                negative_example_and_label = torch.cat((negative_example, torch.zeros([num_negative_example, 1]).to(device)), dim=1)
                boxes = torch.cat((boxes, negative_example_and_label)).to(device)
        if shuffle:
            boxes = boxes[torch.randperm(boxes.size(0))]
        boxes_resized = resize_boxes(boxes[:, 0:4], self.im_info, self.transformed_image_size[0])
        proposals = [boxes_resized]
        with torch.no_grad():
            roi_pool_feat = self.box_roi_pool(fpn_features, proposals, self.im_info).to(device)

        return {'features': roi_pool_feat, 'boxes': boxes[:, 0:4], 'scores': boxes[:, 4]}

    def update_training_set_classification(self, batch_size, additional_dets, fpn_features,
                                           include_previous_frames=False, shuffle=False):
        print('making a classification set')
        training_set_dict = self.generate_training_set_classification(batch_size, additional_dets, fpn_features, shuffle=shuffle)

        if not include_previous_frames:
            self.training_set_classification = IndividualDataset(self.id)
        self.training_set_classification.append_samples(training_set_dict)

    def generate_validation_set_classfication(self, batch_size, additional_dets, fpn_features, shuffle=False):
        return self.generate_training_set_classification(batch_size, additional_dets, fpn_features, shuffle=shuffle)

    def forward_pass_for_classifier_training(self, features, scores, eval=False, return_scores=False):
        if eval:
            self.box_predictor_classification.eval()
            self.box_head_classification.eval()
#        boxes_resized = resize_boxes(boxes[:, 0:4], self.im_info, self.transformed_image_size[0])
#        proposals = [boxes_resized]
#        with torch.no_grad():
#            roi_pool_feat = box_roi_pool(fpn_features, proposals, self.im_info)
        with torch.no_grad():
            feat = self.box_head_classification(features)
        class_logits, _ = self.box_predictor_classification(feat)
        if return_scores:
            pred_scores = F.softmax(class_logits, -1)
            if eval:
                self.box_predictor_classification.train()
                self.box_head_classification.train()
            return pred_scores[:, 1:].squeeze(dim=1).detach()
        loss = F.cross_entropy(class_logits, scores.long())
        if eval:
            self.box_predictor_classification.train()
            self.box_head_classification.train()
        return loss

    def finetune_classification(self, finetuning_config, box_head_classification, box_predictor_classification,
                                early_stopping=False):
        self.training_set_classification.post_process()
        training_set = self.training_set_classification.get_training_set()

        self.box_head_classification = box_head_classification
        self.box_predictor_classification = box_predictor_classification

        self.box_predictor_classification.train()
        self.box_head_classification.train()
        optimizer = torch.optim.Adam(
            list(self.box_predictor_classification.parameters()) + list(self.box_head_classification.parameters()), lr=float(finetuning_config["learning_rate"]) )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=finetuning_config['gamma'])
        dataloader = torch.utils.data.DataLoader(training_set, batch_size=256)

        for i in range(int(finetuning_config["iterations"])):
            for i_sample, sample_batch in enumerate(dataloader):

                optimizer.zero_grad()
                loss = self.forward_pass_for_classifier_training(sample_batch['features'], sample_batch['scores'], eval=False)

                loss.backward()
                optimizer.step()
                scheduler.step()

            if early_stopping or finetuning_config["validate"] or finetuning_config["plot_training_curves"]:
                positive_scores = self.forward_pass_for_classifier_training(sample_batch['features'][sample_batch['scores']==1], sample_batch['scores'], return_scores=True, eval=True)
                negative_scores = self.forward_pass_for_classifier_training(sample_batch['features'][sample_batch['scores']==0], sample_batch['scores'], return_scores=True, eval=True)

            if finetuning_config["plot_training_curves"]:
                positive_scores = positive_scores[:10]
                negative_scores = negative_scores[:10]
                for sample_idx, score in enumerate(positive_scores):
                    self.plotter.plot('score', 'positive {}'.format(sample_idx),
                                 'Scores Evaluation Classifier for Track {}'.format(self.id),
                                 i, score.cpu().numpy(), train_positive=True)  # dark red
                for sample_idx, score in enumerate(negative_scores):
                    self.plotter.plot('score', 'negative {}'.format(sample_idx),
                                 'Scores Evaluation Classifier for Track {}'.format(self.id),
                                 i, score.cpu().numpy())

            if early_stopping and torch.min(positive_scores) - torch.max(negative_scores) > 0.9:
                    break


        self.box_predictor_classification.eval()
        self.box_head_classification.eval()

        # dets = torch.cat((self.pos, additional_dets))
        # print(self.forward_pass(dets, box_roi_pool, fpn_features, scores=True))

    def forward_pass_for_regressor_training(self, boxes, features, bbox_pred_decoder, ground_truth_boxes, eval=False):
        scaled_gt_box = resize_boxes(
            ground_truth_boxes, self.im_info, self.transformed_image_size[0]).squeeze(0)

        if eval:
            self.box_predictor_regression.eval()
            self.box_head_regression.eval()
        boxes_resized = resize_boxes(boxes[:, 0:4], self.im_info, self.transformed_image_size[0])
        proposals = [boxes_resized]
        # with torch.no_grad():
        #     roi_pool_feat = self.box_roi_pool(fpn_features, proposals, self.im_info)
        # Only train the box prediction head
        roi_pool_feat = features
        with torch.no_grad():
            feat = self.box_head_regression(roi_pool_feat)
        _, bbox_pred_offset = self.box_predictor_regression(feat)
        regressed_boxes = bbox_pred_decoder(bbox_pred_offset, proposals)
        regressed_boxes = regressed_boxes[:, 1:].squeeze(dim=1)

        loss = F.mse_loss(scaled_gt_box, regressed_boxes[:, 0:4])
        if eval:
            self.box_predictor_regression.train()
            self.box_head_regression.train()
        return loss

    def finetune_regression(self, finetuning_config, box_head_regression, box_predictor_regression, bbox_pred_decoder):
        training_set = self.training_set_regression.get_upsampled_dataset(1024)
        self.box_head_regression = box_head_regression
        self.box_predictor_regression = box_predictor_regression
        self.box_predictor_regression.train()
        self.box_head_regression.train()
        dataloader = torch.utils.data.DataLoader(training_set, batch_size=512)
        optimizer = torch.optim.Adam(
            list(self.box_predictor_regression.parameters()), lr=float(finetuning_config["learning_rate"]))
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=finetuning_config['gamma'])

        # training_boxes = self.generate_training_set_regression(self.pos,
        #                                                           finetuning_config["max_displacement"],
        #                                                           batch_size=finetuning_config["batch_size"]).to(device)

       # if finetuning_config["validate"]:
       #     if not self.plotter:
       #         self.plotter = VisdomLinePlotter(env_name='training')
       #     validation_boxes = self.generate_training_set_regression(self.pos, finetuning_config["max_displacement"],
       #     finetuning_config["val_batch_size"]).to(device)
        print("Finetuning track {}".format(self.id))
        save_state_box_predictor = FastRCNNPredictor(1024, 2).to(device)
        save_state_box_predictor.load_state_dict(self.box_predictor_regression.state_dict())

        self.checkpoints[0] = [box_head_regression, save_state_box_predictor]

        for i in range(int(finetuning_config["iterations"])):
            for i_sample, sample_batch in enumerate(dataloader):
                if finetuning_config["validation_over_time"]:
                    # if not self.plotter:
                    #     self.plotter = VisdomLinePlotter(env_name='validation_over_time')
                    #     print("Making Plotter")
                    if np.mod(i + 1, finetuning_config["checkpoint_interval"]) == 0:
                        self.box_predictor_regression.eval()
                        save_state_box_predictor = FastRCNNPredictor(1024, 2).to(device)
                        save_state_box_predictor.load_state_dict(self.box_predictor_regression.state_dict())
                        self.checkpoints[i + 1] = [self.box_head_regression, save_state_box_predictor]
                        # input('Checkpoints are the same: {} {}'.format(i+1, Tracker.compare_weights(self.box_predictor_regression, self.checkpoints[0][1])))
                        self.box_predictor_regression.train()

                optimizer.zero_grad()
                loss = self.forward_pass_for_regressor_training(sample_batch['boxes'], sample_batch['features'],
                                                                bbox_pred_decoder, sample_batch['ground_truth_boxes'],
                                                                eval=False)
                loss.backward()
                optimizer.step()
                #scheduler.step()
        self.box_predictor_regression.eval()
        self.box_head_regression.eval()
        self.training_set_classification = torch.tensor([])
        self.training_set_regression.features = [] # only for current experiment where training set is reinitialized
