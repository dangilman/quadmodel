def load_preset_lens(lens_name):
    #
    # if lens_name[0:4] == 'MOCK':
    #     return _load_mock_lens(lens_name)

    if lens_name == 'B1422':
        from quadmodel.data.b1422 import B1422
        return B1422()
    elif lens_name == 'WGD2038':
        from quadmodel.data.wgd2038 import WGD2038
        return WGD2038()
    elif lens_name == 'RXJ0911':
        from quadmodel.data.rxj0911 import RXJ0911
        return RXJ0911()
    elif lens_name == 'HE0435':
        from quadmodel.data.he0435 import HE0435
        return HE0435()
    elif lens_name == 'PSJ1606':
        from quadmodel.data.psj1606 import PSJ1606
        return PSJ1606()
    elif lens_name == 'WFI2026':
        from quadmodel.data.wfi2026 import WFI2026
        return WFI2026()
    elif lens_name == 'PG1115':
        from quadmodel.data.pg1115 import PG1115
        return PG1115()
    elif lens_name == 'WFI2033':
        from quadmodel.data.wfi2033 import WFI2033
        return WFI2033()
    elif lens_name == 'MG0414':
        from quadmodel.data.mg0414 import MG0414
        return MG0414()
    elif lens_name == 'WGDJ0405':
        from quadmodel.data.wgdj0405 import WGDJ0405
        return WGDJ0405()
    elif lens_name == 'RXJ1131':
        from quadmodel.data.rxj1131 import RXJ1131
        return RXJ1131()
    else:
        raise Exception('lens name '+str(lens_name)+' not recognized.')
