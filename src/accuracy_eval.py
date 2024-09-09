import torch


class AccuracyEval:
    def __init__(self):
        self.ious = []
        self.proportions = []

    def __call__(self):
        iou_avg = sum(self.ious) / len(self.ious)
        proportions_avg = sum(self.proportions) / len(self.proportions)

        return iou_avg, proportions_avg

    def add_batch(self, predictions, targets):
        predictions_discrete = (predictions[:, 1, :, :] > predictions[:, 0, :, :]).bool()
        targets_discrete = (targets[:, 0, :, :] > 0).bool()

        intersection = torch.sum(predictions_discrete & targets_discrete).float()
        union = torch.sum(predictions_discrete | targets_discrete).float()

        iou = intersection / (union + 1e-6)
        proportion = intersection / (torch.sum(targets_discrete) + 1e-6)

        self.ious.append(iou)
        self.proportions.append(proportion)


class PixelDistanceEval:
    def __init__(self):
        self.distances = []
        self.inverted_distances = []

    def __call__(self):
        return sum(self.distances) / (len(self.distances) + 1e-6), sum(self.inverted_distances) / (len(self.inverted_distances) + 1e-6)

    def add_batch(self, predictions, targets):
        predictions_discrete = (predictions[:, 1, :, :] > predictions[:, 0, :, :]).bool()
        targets_discrete = (targets[:, 0, :, :] > 0).bool()

        batch_size = predictions.shape[0]

        for i in range(batch_size):
            pred = predictions_discrete[i].squeeze()
            label = targets_discrete[i].squeeze()

            pred_coords = (pred == True).nonzero(as_tuple=False)
            label_coords = (label == True).nonzero(as_tuple=False)

            if pred_coords.size(0) == 0 or label_coords.size(0) == 0:
                continue

            dists = torch.cdist(pred_coords.float(), label_coords.float())
            inverted_dists = torch.cdist(label_coords.float(), pred_coords.float())

            min_dists, _ = torch.min(dists, dim=1)
            min_inverted_dists, _ = torch.min(inverted_dists, dim=1)

            total_distance = torch.sum(min_dists)
            total_samples = min_dists.size(0)

            total_inverted_distance = torch.sum(min_inverted_dists)
            total_inverted_samples = min_inverted_dists.size(0)

            avg_distance = total_distance / (total_samples + 1e-6)
            avg_inverted_distance = total_inverted_distance / (total_inverted_samples + 1e-6)
            self.distances.append(avg_distance)
            self.inverted_distances.append(avg_inverted_distance)
