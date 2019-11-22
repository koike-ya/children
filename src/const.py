
PHASES = ['train', 'val', 'test']
LABELS = {'none': 0, 'seiz': 1, 'arti': 2}
ICTALS_3 = {'interictal': 0, 'preictal': 1, 'ictal': 2}


# children hospital
# 'MJ01128Z'は正例がないため除去
CHILDREN_PATIENTS_1 = ['YJ0112PQ', 'MJ00803P', 'YJ0100DP', 'YJ0100E9', 'MJ00802S', 'YJ01133T', 'YJ0112AU', 'WJ01003H',
                      'WJ010024']
CHILDREN_PATIENTS_2 = ['YJ011412', 'YJ01140M', 'YJ0113ZP', 'YJ0113YL']#'YJ01140Q',
CHILDREN_PATIENTS = CHILDREN_PATIENTS_1# + CHILDREN_PATIENTS_2

# chb-mit
CHANNEL_CHANGED_PATIENTS = ['chb04', 'chb09', 'chb11', 'chb12', 'chb13', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19']
TIME_NOT_LISTED_PATIENTS = ['chb24']
CHBMIT_PATIENTS = list(set([f'chb{i:02}' for i in range(1, 25)]) - set(CHANNEL_CHANGED_PATIENTS) -
                       set(TIME_NOT_LISTED_PATIENTS))
CHBMIT_PATIENTS.sort()
