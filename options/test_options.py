from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results_pixelization/', help='saves results here.')
        self.parser.add_argument('--phase', type=str, default='test', help='train, test')
        self.parser.add_argument('--which_epoch', type=str, default='150', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=10, help='how many test images to run')

        # cartoonGan para
        self.parser.add_argument('--natural_input', default = './input/natural_input')
        self.parser.add_argument('--model_path', default = './cartoonGan/pretrained_model')
        self.parser.add_argument('--style', default = 'Hosoda')
        self.parser.add_argument('--cartoon_output', default = './input/cartoon_input/testA')
        self.parser.add_argument('--gpu', type=int, default = 0)
        
        self.isTrain = False
