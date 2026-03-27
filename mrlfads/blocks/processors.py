import torch
from torch import nn
from typing import Tuple, List


class CommunicatorProcessor(nn.Module):
    def __init__(self, areas):
        super().__init__()
        self.areas = areas
        self.area_names = list(areas.keys())

    def forward(
        self,
        emission_states,
        sample: bool = True,
        mask_hn: bool = False,
        com_noise: tuple = (),
    ):
        com_outputs = []
        com_params = []

        for i, area_name in enumerate(self.area_names):
            area = self.areas[area_name]

            if area.communicator is not None:
                com_samp, com_params_t = area.communicator(
                    emission_states,
                    sample=sample,
                    mask_hn=mask_hn,
                    eps=com_noise[i],
                )
            else:
                com_samp, com_params_t = None, None

            com_outputs.append(com_samp)
            com_params.append(com_params_t)

        return tuple(com_outputs), tuple(com_params)


class DecoderProcessor(nn.Module):
    def __init__(self, areas):
        super().__init__()
        self.areas = areas
        self.area_names = list(areas.keys())

    def forward(
        self,
        inputs_tuple,
        states_tuple,
        sample: bool = True,
        dec_noise: tuple = (),
    ):
        new_states = []
        co_params_all = []
        con_outputs = []

        for i, area_name in enumerate(self.area_names):
            area = self.areas[area_name]
            inputs = inputs_tuple[i]
            states = states_tuple[i]

            new_state, co_params, con_output = area.decoder(
                inputs, states, sample=sample, eps=dec_noise[i],
            )

            new_states.append(new_state)
            co_params_all.append(co_params)
            con_outputs.append(con_output)

        return tuple(new_states), tuple(co_params_all), tuple(con_outputs)


class ReadoutProcessor(nn.Module):
    def __init__(self, areas):
        super().__init__()
        self.areas = areas
        self.area_names = list(areas.keys())

    def forward(
        self,
        new_states_tuple,
        batch_sizes: List[int],
    ):
        outputs = []

        for i, area_name in enumerate(self.area_names):
            area = self.areas[area_name]
            new_state = new_states_tuple[i]

            ahps = area.hparams
            factor_state = new_state[..., -ahps.fac_dim:]
            factor_state_split = torch.split(factor_state, batch_sizes)

            area_outputs = []
            for s_idx in range(len(batch_sizes)):
                rates = area.readout[s_idx](factor_state_split[s_idx])
                area_outputs.append(rates)

            outputs.append(tuple(area_outputs))

        return tuple(outputs)