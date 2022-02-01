def load_preset_lens(lens_name):

    if lens_name[0:4] == 'MOCK':
        return _load_mock_lens(lens_name)

    if lens_name == 'B1422':
        from quadmodel.data.b1422 import B1422
        return B1422()
    elif lens_name == 'WGD2038':
        from quadmodel.data.wgd2038 import WGD2038
        return WGD2038()
    elif lens_name == 'RXJ0911':
        from quadmodel.data.rxj0911 import RXJ0911
        return RXJ0911()
    else:
        raise Exception('lens name '+str(lens_name)+' not recognized.')

def _load_mock_lens(lens_name):

    if lens_name == 'MOCK_1_MIDIR':
        from quadmodel.data.mock_lenses.mock_1 import Mock_1_MIDIR
        return Mock_1_MIDIR()
    elif lens_name == 'MOCK_1_NL':
        from quadmodel.data.mock_lenses.mock_1 import Mock_1_NL
        return Mock_1_NL()

    elif lens_name == 'MOCK_2_MIDIR':
        from quadmodel.data.mock_lenses.mock_2 import Mock_2_MIDIR
        return Mock_2_MIDIR()
    elif lens_name == 'MOCK_2_NL':
        from quadmodel.data.mock_lenses.mock_2 import Mock_2_NL
        return Mock_2_NL()

    elif lens_name == 'MOCK_3_MIDIR':
        from quadmodel.data.mock_lenses.mock_3 import Mock_3_MIDIR
        return Mock_3_MIDIR()
    elif lens_name == 'MOCK_3_NL':
        from quadmodel.data.mock_lenses.mock_3 import Mock_3_NL
        return Mock_3_NL()

    elif lens_name == 'MOCK_4_MIDIR':
        from quadmodel.data.mock_lenses.mock_4 import Mock_4_MIDIR
        return Mock_4_MIDIR()
    elif lens_name == 'MOCK_4_NL':
        from quadmodel.data.mock_lenses.mock_4 import Mock_4_NL
        return Mock_4_NL()

    elif lens_name == 'MOCK_5_MIDIR':
        from quadmodel.data.mock_lenses.mock_5 import Mock_5_MIDIR
        return Mock_5_MIDIR()
    elif lens_name == 'MOCK_5_NL':
        from quadmodel.data.mock_lenses.mock_5 import Mock_5_NL
        return Mock_5_NL()

    elif lens_name == 'MOCK_6_MIDIR':
        from quadmodel.data.mock_lenses.mock_6 import Mock_6_MIDIR
        return Mock_6_MIDIR()
    elif lens_name == 'MOCK_6_NL':
        from quadmodel.data.mock_lenses.mock_6 import Mock_6_NL
        return Mock_6_NL()

    elif lens_name == 'MOCK_7_MIDIR':
        from quadmodel.data.mock_lenses.mock_7 import Mock_7_MIDIR
        return Mock_7_MIDIR()
    elif lens_name == 'MOCK_7_NL':
        from quadmodel.data.mock_lenses.mock_7 import Mock_7_NL
        return Mock_7_NL()

    elif lens_name == 'MOCK_8_MIDIR':
        from quadmodel.data.mock_lenses.mock_8 import Mock_8_MIDIR
        return Mock_8_MIDIR()
    elif lens_name == 'MOCK_8_NL':
        from quadmodel.data.mock_lenses.mock_8 import Mock_8_NL
        return Mock_8_NL()

    elif lens_name == 'MOCK_9_MIDIR':
        from quadmodel.data.mock_lenses.mock_9 import Mock_9_MIDIR
        return Mock_9_MIDIR()
    elif lens_name == 'MOCK_9_NL':
        from quadmodel.data.mock_lenses.mock_9 import Mock_9_NL
        return Mock_9_NL()

    elif lens_name == 'MOCK_10_MIDIR':
        from quadmodel.data.mock_lenses.mock_10 import Mock_10_MIDIR
        return Mock_10_MIDIR()
    elif lens_name == 'MOCK_10_NL':
        from quadmodel.data.mock_lenses.mock_10 import Mock_10_NL
        return Mock_10_NL()
