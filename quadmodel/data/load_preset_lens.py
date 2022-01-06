def load_preset_lens(lens_name):

    if lens_name == 'B1422':
        from quadmodel.data.b1422 import B1422
        return B1422()
    elif lens_name == 'WGD2038':
        from quadmodel.data.wgd2038 import WGD2038
        return WGD2038()
    else:
        raise Exception('lens name '+str(lens_name)+' not recognized.')
