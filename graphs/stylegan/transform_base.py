class TransformGraph():
    def __init__(self, walk_type, nsliders, eps, N_f, img_size, *args, **kwargs):

        # set class vars
        self.Nsliders = nsliders
        self.walk_type = walk_type
        self.N_f = N_f # NN num_steps
        self.eps = eps # NN step_size
        self.img_size = img_size

    def vis_image_batch(self, graph_inputs, filename,
                        batch_start, wgt=False, wmask=False, num_panels=7):
        raise NotImplementedError('Subclass should implement vis_image_batch')

class PixelTransform(TransformGraph):
    def __init__(self, *args, **kwargs):
        TransformGraph.__init__(self, *args, **kwargs)

    def get_distribution_statistic(self, img, channel=None):
        raise NotImplementedError('Subclass should implement get_distribution_statistic')

class BboxTransform(TransformGraph):
    def __init__(self, *args, **kwargs):
        TransformGraph.__init__(self, *args, **kwargs)

    def get_distribution_statistic(self, img, channel=None):
        raise NotImplementedError('Subclass should implement get_distribution_statistic')