from torch import optim

from module.LIF_Module import RMFCLIF, RMConvLIF
from module.RM import *


def create_net(config):
    # define approximate firing function
    class ActFun(torch.autograd.Function):
        def __init__(self):
            super(ActFun, self).__init__()

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return input.ge(0.0).float()

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            temp = abs(input) < config.lens
            return grad_output * temp.float() / (2 * config.lens)

    # cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
    cfg_cnn = [
        (
            2,
            64,
            1,
            1,
            3,
        ),
        (
            64,
            128,
            1,
            1,
            3,
        ),
        (
            128,
            128,
            1,
            1,
            3,
        ),
    ]
    # pooling kernel_size
    cfg_pool = [1, 2, 2]
    # fc layer
    cfg_fc = [cfg_cnn[2][1] * 8 * 8, 256, config.target_size]

    # Net
    class Net(nn.Module):
        def __init__(
            self,
        ):
            super(Net, self).__init__()
            h, w = config.im_height, config.im_width
            in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
            pooling_kernel_size = cfg_pool[0]
            h, w = h // cfg_pool[0], w // cfg_pool[0]
            self.convAttLIF0 = RMConvLIF(
                p=0,
                attention=config.attention,
                inputSize=in_planes,
                hiddenSize=out_planes,
                kernel_size=(kernel_size, kernel_size),
                spikeActFun=ActFun.apply,
                init_method=config.init_method,
                useBatchNorm=True,
                pooling_kernel_size=pooling_kernel_size,
                T=config.T,
                pa_dict={
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "Vreset": config.Vreset,
                    "Vthres": config.Vthres,
                },
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
                RM=config.RM,
                c_sparsity_ratio=config.c_sparsity_ratio,
                t_sparsity_ratio=config.t_sparsity_ratio,
                reserve_coefficient=config.reserve_coefficient,
            )

            in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
            pooling_kernel_size = cfg_pool[1]
            h, w = h // cfg_pool[1], w // cfg_pool[1]
            self.convAttLIF1 = RMConvLIF(
                p=0,
                attention=config.attention,
                inputSize=in_planes,
                hiddenSize=out_planes,
                kernel_size=(kernel_size, kernel_size),
                spikeActFun=ActFun.apply,
                init_method=config.init_method,
                useBatchNorm=True,
                pooling_kernel_size=pooling_kernel_size,
                T=config.T,
                pa_dict={
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "Vreset": config.Vreset,
                    "Vthres": config.Vthres,
                },
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
                RM=config.RM,
                c_sparsity_ratio=config.c_sparsity_ratio,
                t_sparsity_ratio=config.t_sparsity_ratio,
                reserve_coefficient=config.reserve_coefficient,
            )

            in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[2]
            pooling_kernel_size = cfg_pool[2]
            h, w = h // cfg_pool[2], w // cfg_pool[2]
            self.convAttLIF2 = RMConvLIF(
                p=0,
                attention=config.attention,
                inputSize=in_planes,
                hiddenSize=out_planes,
                kernel_size=(kernel_size, kernel_size),
                spikeActFun=ActFun.apply,
                init_method=config.init_method,
                useBatchNorm=True,
                pooling_kernel_size=pooling_kernel_size,
                T=config.T,
                pa_dict={
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "Vreset": config.Vreset,
                    "Vthres": config.Vthres,
                },
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
                RM=config.RM,
                c_sparsity_ratio=config.c_sparsity_ratio,
                t_sparsity_ratio=config.t_sparsity_ratio,
                reserve_coefficient=config.reserve_coefficient,
            )

            self.FC0 = RMFCLIF(
                p=0,
                attention="no" if config.attention == "no" else "T",
                inputSize=cfg_fc[0],
                hiddenSize=cfg_fc[1],
                spikeActFun=ActFun.apply,
                useBatchNorm=True,
                T=config.T,
                pa_dict={
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "Vreset": config.Vreset,
                    "Vthres": config.Vthres,
                },
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
            )

            self.FC1 = RMFCLIF(
                attention="no" if config.attention == "no" else "T",
                inputSize=cfg_fc[1],
                hiddenSize=cfg_fc[2],
                spikeActFun=ActFun.apply,
                useBatchNorm=True,
                T=config.T,
                pa_dict={
                    "alpha": config.alpha,
                    "beta": config.beta,
                    "Vreset": config.Vreset,
                    "Vthres": config.Vthres,
                },
                track_running_stats=config.track_running_stats,
                mode_select=config.mode_select,
                mem_act=config.mem_act,
                TR_model=config.TR_model,
            )

        def forward(self, input):
            b, t, _, _, _ = input.size()

            outputs = input.permute(1, 0, 2, 3, 4)
            pred_saliency_list = []

            outputs, pred_saliency = self.convAttLIF0(outputs)
            pred_saliency_list.append(pred_saliency)

            outputs, pred_saliency = self.convAttLIF1(outputs)
            pred_saliency_list.append(pred_saliency)

            outputs, pred_saliency = self.convAttLIF2(outputs)
            pred_saliency_list.append(pred_saliency)

            outputs = outputs.reshape(t, b, -1)

            outputs = self.FC0(outputs)
            outputs = self.FC1(outputs)
            outputs = torch.sum(outputs.permute(1, 0, 2), dim=1)
            outputs = outputs / t

            lasso = 0
            for i in range(len(pred_saliency_list)):
                if pred_saliency_list[i] is not None:
                    lasso += pred_saliency_list[i].abs().sum()
                else:
                    lasso += 0
            return outputs, lasso

    config.model = Net().to(config.device)

    # optimizer
    config.optimizer = optim.Adam(
        config.model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    config.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=config.optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )
    config.model = nn.DataParallel(config.model, device_ids=config.device_ids)
