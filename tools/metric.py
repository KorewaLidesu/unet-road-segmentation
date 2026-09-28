import numpy as np
from skimage.morphology import skeletonize
import networkx as nx
import sknw
from scipy.spatial import cKDTree

class Evaluator(object):

    def __init__(self, num_class, topology=True):
        self.num_class = num_class
        # kaggle: topology=False skips skeletonize+sknw+KDTree per image (clDice/APLS -> nan).
        self.topology = topology
        self.confusion_matrices = []
        self.apls_scores = []
        
        self.cldice_tprec_num = 0.0
        self.cldice_tprec_den = 0.0
        self.cldice_trec_num = 0.0
        self.cldice_trec_den = 0.0

    def reset(self):
        self.confusion_matrices = []
        self.apls_scores = []
        
        self.cldice_tprec_num = 0.0
        self.cldice_tprec_den = 0.0
        self.cldice_trec_num = 0.0
        self.cldice_trec_den = 0.0

    def _skeletonize(self, mask):

        binary_mask = (mask == 0).astype(np.uint8)
        skel = skeletonize(binary_mask.astype(bool))
        return skel.astype(np.uint8)

    def _compute_apls(self, skel_gt, skel_pred, num_samples=50):
        if np.sum(skel_gt) == 0 and np.sum(skel_pred) == 0:
            return 1.0
        if np.sum(skel_gt) == 0 or np.sum(skel_pred) == 0:
            return 0.0

        graph_gt = sknw.build_sknw(skel_gt)
        graph_pred = sknw.build_sknw(skel_pred)

        if len(graph_gt.nodes()) < 2 or len(graph_pred.nodes()) < 2:
            return 0.0
            
        def compute_one_way(G_source, G_target, num_samples):
            nodes_source = np.array([G_source.nodes[n]['o'] for n in G_source.nodes()])
            nodes_target = np.array([G_target.nodes[n]['o'] for n in G_target.nodes()])
            target_node_list = list(G_target.nodes())
            source_node_list = list(G_source.nodes())

            tree_target = cKDTree(nodes_target)

            if len(source_node_list) > num_samples:
                step = max(1, len(source_node_list) // num_samples)
                sampled_nodes = source_node_list[::step][:num_samples]
            else:
                sampled_nodes = source_node_list

            score = 0.0
            valid_pairs = 0

            for i in range(len(sampled_nodes)):
                for j in range(i + 1, len(sampled_nodes)):
                    n_i_src, n_j_src = sampled_nodes[i], sampled_nodes[j]

                    try:
                        L_src = nx.shortest_path_length(G_source, source=n_i_src, target=n_j_src, weight='weight')
                    except nx.NetworkXNoPath:
                        continue

                    _, n_i_tgt_idx = tree_target.query(nodes_source[source_node_list.index(n_i_src)])
                    _, n_j_tgt_idx = tree_target.query(nodes_source[source_node_list.index(n_j_src)])
                    n_i_tgt = target_node_list[n_i_tgt_idx]
                    n_j_tgt = target_node_list[n_j_tgt_idx]

                    try:
                        L_tgt = nx.shortest_path_length(G_target, source=n_i_tgt, target=n_j_tgt, weight='weight')
                    except nx.NetworkXNoPath:
                        L_tgt = float('inf')

                    if L_src > 0:
                        diff = abs(L_src - L_tgt) / L_src
                        score += (1.0 - min(1.0, diff))
                    valid_pairs += 1

            if valid_pairs == 0:
                return 0.0
            return score / valid_pairs

        score_gt2pred = compute_one_way(graph_gt, graph_pred, num_samples)
        score_pred2gt = compute_one_way(graph_pred, graph_gt, num_samples)

        return (score_gt2pred + score_pred2gt) / 2.0

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape, gt_image.shape)
        
        mask_valid = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask_valid].astype('int') + pre_image[mask_valid]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        self.confusion_matrices.append(confusion_matrix)

        if not self.topology:
            return

        mask_gt = (gt_image == 0).astype(np.uint8)
        mask_pred = (pre_image == 0).astype(np.uint8)

        skel_gt = self._skeletonize(gt_image)
        skel_pred = self._skeletonize(pre_image)

        self.cldice_tprec_num += np.sum(skel_pred * mask_gt)
        self.cldice_tprec_den += np.sum(skel_pred)

        self.cldice_trec_num += np.sum(skel_gt * mask_pred)
        self.cldice_trec_den += np.sum(skel_gt)

        try:
            apls_score = self._compute_apls(skel_gt, skel_pred)
            self.apls_scores.append(apls_score)
        except Exception:
            self.apls_scores.append(0.0)

    def clDice(self):
        if not self.topology:
            return float('nan')
        if self.cldice_tprec_den == 0:
            tprec = 1.0 if self.cldice_trec_den == 0 else 0.0
        else:
            tprec = self.cldice_tprec_num / self.cldice_tprec_den

        if self.cldice_trec_den == 0:
            trec = 1.0 if self.cldice_tprec_den == 0 else 0.0
        else:
            trec = self.cldice_trec_num / self.cldice_trec_den

        if tprec + trec == 0:
            return 0.0
        return 2.0 * tprec * trec / (tprec + trec)

    def APLS(self):
        if not self.topology:
            return float('nan')
        if not self.apls_scores:
            return 0.0
        return np.mean(self.apls_scores)

    def Intersection_over_Union(self):
        matrix = np.sum(self.confusion_matrices, axis=0)
        tp = np.diag(matrix)
        fn = matrix.sum(axis=1) - tp  
        fp = matrix.sum(axis=0) - tp  
        return tp / (tp + fn + fp + 1e-10)

    def F1(self):
        matrix = np.sum(self.confusion_matrices, axis=0)
        tp = np.diag(matrix)
        fn = matrix.sum(axis=1) - tp
        fp = matrix.sum(axis=0) - tp
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        return (2.0 * precision * recall) / (precision + recall + 1e-10)

    def Precision(self):
        matrix = np.sum(self.confusion_matrices, axis=0)
        tp = np.diag(matrix)
        fp = matrix.sum(axis=0) - tp
        return tp / (tp + fp + 1e-10)

    def Recall(self):
        matrix = np.sum(self.confusion_matrices, axis=0)
        tp = np.diag(matrix)
        fn = matrix.sum(axis=1) - tp
        return tp / (tp + fn + 1e-10)

    def OA(self):
        matrix = np.sum(self.confusion_matrices, axis=0)
        return np.diag(matrix).sum() / (matrix.sum() + 1e-10)

    def confusion_matrix(self):
        """Summed confusion matrix, or zeros when nothing has been added yet."""
        if not self.confusion_matrices:
            return np.zeros((self.num_class, self.num_class), dtype=np.int64)
        return np.sum(self.confusion_matrices, axis=0)

    def add_confusion_matrix(self, matrix):
        """Merge a confusion matrix computed elsewhere (e.g. another DDP rank)."""
        matrix = np.asarray(matrix)
        assert matrix.shape == (self.num_class, self.num_class), matrix.shape
        self.confusion_matrices.append(matrix)

if __name__ == '__main__':
    gt = np.array([[0, 1, 1],
                   [1, 1, 1],
                   [1, 0, 1]])

    pre = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [1, 1, 1]])

    eval = Evaluator(num_class=2)
    eval.add_batch(gt, pre)
    
    print("F1:", eval.F1())
    print("IoU:", eval.Intersection_over_Union())
    print("clDice:", eval.clDice())
    print("APLS:", eval.APLS())