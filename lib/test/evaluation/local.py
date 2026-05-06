from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    project_root = "E:\\biyesheji\\SUTrack-main11"

    settings.davis_dir = ''
    settings.got10k_lmdb_path = f'{project_root}/data/got10k_lmdb'
    settings.got10k_path = f'{project_root}/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_extension_subset_path = f'{project_root}/data/lasot_extension_subset'
    settings.lasot_lmdb_path = f'{project_root}/data/lasot_lmdb'
    settings.lasot_path = f'{project_root}/data/lasot'
    settings.lasotlang_path = f'{project_root}/data/lasot'
    settings.network_path = f'{project_root}/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = f'{project_root}/data/nfs'
    settings.otb_path = f'{project_root}/data/OTB2015'
    settings.otblang_path = f'{project_root}/data/otb_lang'
    settings.prj_dir = project_root
    settings.result_plot_path = f'{project_root}/test/result_plots'
    settings.results_path = f'{project_root}/test/tracking_results'    # Where to store tracking results
    settings.save_dir = project_root
    settings.segmentation_path = f'{project_root}/test/segmentation_results'
    settings.tc128_path = f'{project_root}/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = f'{project_root}/data/tnl2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = f'{project_root}/data/trackingnet'
    settings.uav_path = f'{project_root}/data/UAV123'
    settings.vot_path = f'{project_root}/data/VOT2019'
    settings.youtubevos_dir = ''
    settings.antiuav_path = 'E:\\biyesheji\\SUTrack-main11\\data\\AntI-UAV'

    return settings
