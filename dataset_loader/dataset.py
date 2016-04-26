import os
import numpy as np
import pandas as pd
from datetime import date, datetime
from joblib import Memory
from sklearn.datasets.base import Bunch
from utils import (_get_data_base_dir, _rid_to_ptid, _get_dx,
                   _get_cache_base_dir, _glob_subject_img, _ptid_to_rid,
                   _get_group_indices, _get_subjects_and_description,
                   _get_vcodes, _get_dob, _get_gender, _get_mmse,
                   _get_cdr, _get_gdscale, _get_faq, _get_npiq,
                   _get_adas, _get_nss, _get_neurobat)


DX_LIST = np.array(['None',
                    'Normal',
                    'MCI',
                    'AD',
                    'Normal->MCI',
                    'MCI->AD',
                    'Normal->AD',
                    'MCI->Normal',
                    'AD->MCI',
                    'AD->Normal'])


def load_adni_longitudinal_mmse_score():
    """ Returns longitudinal mmse scores
    """
    BASE_DIR = _get_data_base_dir('ADNI_csv')
    roster = pd.read_csv(os.path.join(BASE_DIR, 'ROSTER.csv'))
    dx = pd.read_csv(os.path.join(BASE_DIR, 'DXSUM_PDXCONV_ADNIALL.csv'))
    fs = pd.read_csv(os.path.join(BASE_DIR, 'MMSE.csv'))

    # extract nans free mmse
    mmse = fs['MMSCORE'].values
    idx_num = fs['MMSCORE'].notnull().values
    mmse = mmse[idx_num]

    # extract roster id
    rids = fs['RID'].values[idx_num]

    # caching dataframe extraction functions
    CACHE_DIR = _get_cache_base_dir()
    cache_dir = os.path.join(CACHE_DIR, 'joblib', 'load_data_cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    memory = Memory(cachedir=cache_dir, verbose=0)

    def _getptidsmmse(rids):
        return [_rid_to_ptid(rid, roster) for rid in rids]

    # get subject id
    ptids = memory.cache(_getptidsmmse)(rids)
    # extract visit code (don't use EXAMDATE ; null for GO/2)
    vcodes = fs['VISCODE'].values
    vcodes = vcodes[idx_num]
    vcodes2 = fs['VISCODE2'].values
    vcodes2 = vcodes2[idx_num]

    def _getdxmmse(rids, vcodes2):
        return map(lambda x, y: DX_LIST[_get_dx(x, dx, viscode=y)],
                   rids, vcodes2)

    # get diagnosis
    dx_group = memory.cache(_getdxmmse)(rids, vcodes2)

    return Bunch(dx_group=np.array(dx_group), subjects=np.array(ptids),
                 mmse=mmse, exam_codes=vcodes, exam_codes2=vcodes2)


def load_adni_longitudinal_csf_biomarker():
    """ Returns longitudinal csf measures
    """
    BASE_DIR = _get_data_base_dir('ADNI_csv')
    roster = pd.read_csv(os.path.join(BASE_DIR, 'ROSTER.csv'))
    dx = pd.read_csv(os.path.join(BASE_DIR, 'DXSUM_PDXCONV_ADNIALL.csv'))
    csf_files = ['UPENNBIOMK.csv', 'UPENNBIOMK2.csv', 'UPENNBIOMK3.csv',
                 'UPENNBIOMK4_09_06_12.csv', 'UPENNBIOMK5_10_31_13.csv',
                 'UPENNBIOMK6_07_02_13.csv', 'UPENNBIOMK7.csv',
                 'UPENNBIOMK8.csv']
    cols = ['RID', 'VISCODE', 'ABETA', 'PTAU', 'TAU']
    # 3,4,5,7,8
    csf = pd.DataFrame()
    for csf_file in csf_files[2:]:
        fs = pd.read_csv(os.path.join(BASE_DIR, csf_file))
        csf = csf.append(fs[cols])

    # remove nans from csf values
    biom = csf[cols[2:]].values
    idx = np.array([~np.isnan(v).any() for v in biom])
    biom = biom[idx]
    # get phenotype
    vcodes = csf['VISCODE'].values[idx]
    rids = csf['RID'].values[idx]

    # caching dataframe extraction functions
    CACHE_DIR = _get_cache_base_dir()
    cache_dir = os.path.join(CACHE_DIR, 'joblib', 'load_data_cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    memory = Memory(cachedir=cache_dir, verbose=0)

    def _getptidscsf(rids):
        return map(lambda x: _rid_to_ptid(x, roster), rids)
    ptids = memory.cache(_getptidscsf)(rids)

    # get diagnosis
    def _getdxcsf(rids, vcodes):
        return map(lambda x, y: DX_LIST[_get_dx(x, dx, viscode=y)],
                   rids, vcodes)
    dx_group = memory.cache(_getdxcsf)(rids, vcodes)

    return Bunch(dx_group=np.array(dx_group), subjects=np.array(ptids),
                 csf=np.array(biom), exam_codes=np.array(vcodes),
                 exam_codes2=np.array(vcodes))


def load_adni_longitudinal_hippocampus_volume():
    """ Returns longitudinal hippocampus measures
    """

    BASE_DIR = _get_data_base_dir('ADNI_csv')

    roster = pd.read_csv(os.path.join(BASE_DIR, 'ROSTER.csv'))
    dx = pd.read_csv(os.path.join(BASE_DIR, 'DXSUM_PDXCONV_ADNIALL.csv'))
    fs = pd.read_csv(os.path.join(BASE_DIR, 'UCSFFSX51_05_20_15.csv'))

    # extract hippocampus numerical values
    column_idx = np.arange(131, 147)
    cols = ['ST' + str(c) + 'HS' for c in column_idx]
    hipp = fs[cols].values
    idx_num = np.array([~np.isnan(h).all() for h in hipp])
    hipp = hipp[idx_num, :]

    # extract roster id
    rids = fs['RID'].values[idx_num]

    # caching dataframe extraction functions
    CACHE_DIR = _get_cache_base_dir()
    cache_dir = os.path.join(CACHE_DIR, 'joblib', 'load_data_cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    memory = Memory(cachedir=cache_dir, verbose=0)

    # get subject id
    def _getptidshippo(rids):
        return [_rid_to_ptid(rid, roster) for rid in rids]
    ptids = memory.cache(_getptidshippo)(rids)

    # extract exam date
    exams = fs['EXAMDATE'].values[idx_num]
    vcodes = fs['VISCODE'].values[idx_num]
    vcodes2 = fs['VISCODE2'].values[idx_num]
    exams = map(lambda e: date(int(e[:4]), int(e[5:7]), int(e[8:])), exams)
    exams = np.array(exams)

    # extract diagnosis
    def _getdxhippo(rids, exams):
        return np.array(map(_get_dx, rids, [dx]*len(rids), exams))
    dx_ind = memory.cache(_getdxhippo)(rids, exams)
    dx_group = DX_LIST[dx_ind]

    return Bunch(dx_group=np.array(dx_group), subjects=np.array(ptids),
                 hipp=np.array(hipp), exam_dates=np.array(exams),
                 exam_codes=np.array(vcodes), exam_codes2=np.array(vcodes2))


def load_adni_longitudinal_rs_fmri_DARTEL():
    """ Returns longitudinal func processed with DARTEL
    """
    return load_adni_longitudinal_rs_fmri('ADNI_longitudinal_rs_fmri_DARTEL',
                                          'resampled*.nii')


def load_adni_longitudinal_rs_fmri(dirname='ADNI_longitudinal_rs_fmri',
                                   prefix='wr*.nii'):
    """ Returns paths of ADNI rs-fMRI
    """

    # get file paths and description
    images, subject_paths, description = _get_subjects_and_description(
        base_dir=dirname, prefix='I[0-9]*')
    images = np.array(images)
    # get func files
    func_files = map(lambda x: _glob_subject_img(x, suffix='func/' + prefix,
                                                 first_img=True),
                     subject_paths)
    func_files = np.array(func_files)

    # get motion files
    # motions = None
    # motions = map(lambda x: _glob_subject_img(x, suffix='func/' + 'rp_*.txt',
    # first_img=True), subject_paths)

    # get phenotype from csv
    dx = pd.read_csv(os.path.join(_get_data_base_dir('ADNI_csv'),
                                  'DXSUM_PDXCONV_ADNIALL.csv'))
    roster = pd.read_csv(os.path.join(_get_data_base_dir('ADNI_csv'),
                                      'ROSTER.csv'))
    df = description[description['Image_ID'].isin(images)]
    df = df.sort_values(by='Image_ID')
    dx_group = np.array(df['DX_Group'])
    subjects = np.array(df['Subject_ID'])
    exams = np.array(df['EXAM_DATE'])
    exams = map(lambda e: date(int(e[:4]), int(e[5:7]), int(e[8:])), exams)

    # caching dataframe extraction functions
    CACHE_DIR = _get_cache_base_dir()
    cache_dir = os.path.join(CACHE_DIR, 'joblib', 'load_data_cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    memory = Memory(cachedir=cache_dir, verbose=0)

    def _get_ridsfmri(subjects):
        return map(lambda s: _ptid_to_rid(s, roster), subjects)
    rids = np.array(memory.cache(_get_ridsfmri)(subjects))

    def _get_examdatesfmri(rids):
        return map(lambda i: _get_dx(rids[i],
                                     dx, exams[i],
                                     viscode=None,
                                     return_code=True), range(len(rids)))
    exam_dates = np.array(memory.cache(_get_examdatesfmri)(rids))

    def _get_viscodesfmri(rids):
        return map(lambda i: _get_vcodes(rids[i], str(exam_dates[i]), dx),
                   range(len(rids)))
    viscodes = np.array(memory.cache(_get_viscodesfmri)(rids))
    vcodes, vcodes2 = viscodes[:, 0], viscodes[:, 1]

    return Bunch(func=func_files, dx_group=dx_group, exam_codes=vcodes,
                 exam_dates=exam_dates, exam_codes2=vcodes2,
                 subjects=subjects, images=images)


def load_adni_rs_fmri():
    """ Returns paths of ADNI resting-state fMRI
    """

    # get file paths and description
    subjects, subject_paths, description = _get_subjects_and_description(
        base_dir='ADNI_baseline_rs_fmri_mri', prefix='s[0-9]*')
    # get the correct subject_id
    subjects = [s[1:] for s in subjects]

    # get func files
    func_files = map(lambda x: _glob_subject_img(x, suffix='func/swr*.nii',
                                                 first_img=True),
                     subject_paths)

    # get phenotype from csv
    df = description[description['Subject_ID'].isin(subjects)]
    dx_group = np.array(df['DX_Group_x'])
    mmscores = np.array(df['MMSCORE'])

    return Bunch(func=func_files, dx_group=dx_group,
                 mmscores=mmscores, subjects=subjects)


def load_adni_longitudinal_av45_pet():
    """Returns paths of longitudinal ADNI AV45-PET
    """

    # get file paths and description
    (subjects,
     subject_paths,
     description) = _get_subjects_and_description(base_dir='ADNI_av45_pet',
                                                  prefix='I[0-9]*')

    # get pet files
    pet_files = map(lambda x: _glob_subject_img(x, suffix='pet/wr*.nii',
                                                first_img=False),
                    subject_paths)
    idx = [0]
    pet_files_all = []
    for pet_file in pet_files:
        idx.append(idx[-1] + len(pet_file))
        pet_files_all.extend(pet_file)
    pet_files_all = np.array(pet_files_all)

    images = [os.path.split(pet_file)[-1].split('_')[-1][:-4]
              for pet_file in pet_files_all]
    images = np.array(images)

    # get phenotype from csv
    dx = pd.read_csv(os.path.join(_get_data_base_dir('ADNI_csv'),
                                  'DXSUM_PDXCONV_ADNIALL.csv'))
    roster = pd.read_csv(os.path.join(_get_data_base_dir('ADNI_csv'),
                                      'ROSTER.csv'))
    df = description[description['Image_ID'].isin(images)]
    dx_group_all = np.array(df['DX_Group'])
    subjects_all = np.array(df['Subject_ID'])
    ages = np.array(df['Age'])

    exams = np.array(df['Study_Date'])
    exams = map(lambda e: datetime.strptime(e, '%m/%d/%Y').date(), exams)

    # caching dataframe extraction functions
    CACHE_DIR = _get_cache_base_dir()
    cache_dir = os.path.join(CACHE_DIR, 'joblib', 'load_data_cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    memory = Memory(cachedir=cache_dir, verbose=0)

    def _get_ridspet(subjects_all):
        return map(lambda s: _ptid_to_rid(s, roster), subjects_all)
    rids = memory.cache(_get_ridspet)(subjects_all)

    def _get_examdatespet(rids):
        return map(lambda i: _get_dx(rids[i],
                                     dx, exams[i],
                                     viscode=None,
                                     return_code=True), range(len(rids)))
    exam_dates = np.array(memory.cache(_get_examdatespet)(rids))

    def _get_viscodespet(rids):
        return map(lambda i: _get_vcodes(rids[i], str(exam_dates[i]), dx),
                   range(len(rids)))
    viscodes = np.array(memory.cache(_get_viscodespet)(rids))
    if len(viscodes) > 0:
        vcodes, vcodes2 = viscodes[:, 0], viscodes[:, 1]
    else:
        vcodes, vcodes2 = None, None

    return Bunch(pet=pet_files_all,
                 dx_group=dx_group_all,
                 images=images, ages=ages, subjects=subjects_all,
                 exam_codes=vcodes, exam_dates=exam_dates, exam_codes2=vcodes2)


def load_adni_longitudinal_fdg_pet():
    """Returns paths of longitudinal ADNI FDG-PET
    """

    # get file paths and description
    (subjects, subject_paths, description) = _get_subjects_and_description(
        base_dir='ADNI_longitudinal_fdg_pet', prefix='[0-9]*')

    # get pet files
    pet_files = map(lambda x: _glob_subject_img(x, suffix='pet/wr*.nii',
                                                first_img=False),
                    subject_paths)
    idx = [0]
    pet_files_all = []
    for pet_file in pet_files:
        idx.append(idx[-1] + len(pet_file))
        pet_files_all.extend(pet_file)
    pet_files_all = np.array(pet_files_all)

    images = [os.path.split(pet_file)[-1].split('_')[-1][:-4]
              for pet_file in pet_files_all]
    images = np.array(images)

    # get phenotype from csv
    dx = pd.read_csv(os.path.join(_get_data_base_dir('ADNI_csv'),
                                  'DXSUM_PDXCONV_ADNIALL.csv'))
    roster = pd.read_csv(os.path.join(_get_data_base_dir('ADNI_csv'),
                                      'ROSTER.csv'))
    df = description[description['Image_ID'].isin(images)]
    dx_group_all = np.array(df['DX_Group'])
    dx_conv_all = np.array(df['DX_Conv'])
    subjects_all = np.array(df['Subject_ID'])
    ages = np.array(df['Age'])

    exams = np.array(df['Exam_Date'])
    exams = map(lambda e: date(int(e[:4]), int(e[5:7]), int(e[8:])), exams)

    # caching dataframe extraction functions
    CACHE_DIR = _get_cache_base_dir()
    cache_dir = os.path.join(CACHE_DIR, 'joblib', 'load_data_cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    memory = Memory(cachedir=cache_dir, verbose=0)

    def _get_ridspet(subjects_all):
        return map(lambda s: _ptid_to_rid(s, roster), subjects_all)
    rids = memory.cache(_get_ridspet)(subjects_all)

    def _get_examdatespet(rids):
        return map(lambda i: _get_dx(rids[i],
                                     dx, exams[i],
                                     viscode=None,
                                     return_code=True), range(len(rids)))
    exam_dates = np.array(memory.cache(_get_examdatespet)(rids))

    def _get_viscodespet(rids):
        return map(lambda i: _get_vcodes(rids[i], str(exam_dates[i]), dx),
                   range(len(rids)))
    viscodes = np.array(memory.cache(_get_viscodespet)(rids))
    vcodes, vcodes2 = viscodes[:, 0], viscodes[:, 1]

    return Bunch(pet=pet_files_all,
                 dx_group=dx_group_all, dx_conv=dx_conv_all,
                 images=images, ages=ages, subjects=subjects_all,
                 exam_codes=vcodes, exam_dates=exam_dates, exam_codes2=vcodes2)


def load_adni_baseline_rs_fmri():
    """ Returns paths of ADNI rs-fMRI
    """

    # get file paths and description
    (subjects, subject_paths, description) = _get_subjects_and_description(
        base_dir='ADNI_baseline_rs_fmri', prefix='[0-9]*')

    # get func files
    func_files = map(lambda x: _glob_subject_img(x, suffix='func/wr*.nii',
                                                 first_img=True),
                     subject_paths)

    # get phenotype from csv
    df = description[description['Subject_ID'].isin(subjects)]
    dx_group = np.array(df['DX_Group'])

    return Bunch(func=func_files, dx_group=dx_group, subjects=subjects)


def load_adni_rs_fmri_conn(filename):
    """Returns paths of ADNI rs-fMRI processed connectivity
    for a given npy file with shape : n_subjects x n_voxels x n_rois
    """

    FEAT_DIR = _get_data_base_dir('features')
    conn_file = os.path.join(FEAT_DIR, 'smooth_preproc', filename)
    if not os.path.isfile(conn_file):
        raise OSError('Connectivity file not found ...')
    dataset = load_adni_petmr()
    subj_list = dataset['subjects']

    return Bunch(fmri_data=conn_file,
                 dx_group=np.array(dataset['dx_group']),
                 mmscores=np.array(dataset['mmscores']),
                 subjects=subj_list)


def load_adni_fdg_pet():
    """Returns paths of ADNI baseline FDG-PET
    """

    # get file paths and description
    subjects, subject_paths, description = _get_subjects_and_description(
        base_dir='ADNI_baseline_fdg_pet', prefix='s[0-9]*')

    # get the correct subject_id
    subjects = [s[1:] for s in subjects]

    # get pet files
    pet_files = map(lambda x: _glob_subject_img(x, suffix='pet/w*.nii',
                                                first_img=True), subject_paths)
    # get phenotype from csv
    df = description[description['Subject_ID'].isin(subjects)]
    dx_group = np.array(df['DX_Group'])
    mmscores = np.array(df['MMSCORE'])

    return Bunch(pet=pet_files, dx_group=dx_group,
                 mmscores=mmscores, subjects=subjects)


def load_adni_fdg_pet_diff():
    """Returns paths of the diff between PET and fMRI datasets
    """
    pet_dataset = load_adni_fdg_pet()
    fmri_dataset = load_adni_rs_fmri()
    remaining_subjects = np.setdiff1d(pet_dataset['subjects'],
                                      fmri_dataset['subjects'])
    pet_idx = []
    for pet_subject in remaining_subjects:
        pet_idx.append(
            np.where(np.array(pet_dataset['subjects']) == pet_subject)[0][0])
    pet_idx = np.array(pet_idx, dtype=np.intp)
    pet_groups = np.array(pet_dataset['dx_group'])
    pet_groups = pet_groups[pet_idx]
    pet_mmscores = np.array(pet_dataset['mmscores'])
    pet_mmscores = pet_mmscores[pet_idx]
    pet_files = np.array(pet_dataset['pet'])[pet_idx]

    return Bunch(pet=pet_files, dx_group=pet_groups,
                 mmscores=pet_mmscores, subjects=remaining_subjects)


def load_adni_petmr():
    """Returns paths of the intersection between PET and FMRI datasets
    """
    pet_dataset = load_adni_fdg_pet()
    fmri_dataset = load_adni_rs_fmri()
    petmr_subjects = np.intersect1d(pet_dataset['subjects'],
                                    fmri_dataset['subjects'],
                                    assume_unique=True)
    petmr_idx = []
    mrpet_idx = []
    for petmr_subject in petmr_subjects:
        petmr_idx.append(
            np.where(
                np.array(pet_dataset['subjects']) == petmr_subject)[0][0])
        mrpet_idx.append(
            np.where(
                np.array(fmri_dataset['subjects']) == petmr_subject)[0][0])

    petmr_idx = np.array(petmr_idx, dtype=np.intp)
    mrpet_idx = np.array(mrpet_idx, dtype=np.intp)
    pet_groups = np.array(pet_dataset['dx_group'])
    petmr_groups = pet_groups[petmr_idx]
    pet_mmscores = np.array(pet_dataset['mmscores'])
    petmr_mmscores = pet_mmscores[petmr_idx]
    func_files = np.array(fmri_dataset['func'])[mrpet_idx]
    pet_files = np.array(pet_dataset['pet'])[petmr_idx]

    return Bunch(func=func_files, pet=pet_files, dx_group=petmr_groups,
                 mmscores=petmr_mmscores, subjects=petmr_subjects)


def load_adni_masks():
    """Returns paths of masks (pet, fmri, both)

    Returns
    -------
    mask : Bunch containing:
           - mask_pet
           - mask_fmri
           - mask_pet_longitudinal
           - mask_petmr
    """
    BASE_DIR = _get_data_base_dir('features/masks')

    return Bunch(pet=os.path.join(BASE_DIR, 'mask_pet.nii.gz'),
                 fmri=os.path.join(BASE_DIR, 'mask_fmri.nii.gz'),
                 pet_longitudinal=os.path.join(BASE_DIR,
                                               'mask_longitudinal_fdg_pet'
                                               '.nii.gz'),
                 petmr=os.path.join(BASE_DIR, 'mask_petmr.nii.gz'),
                 petmr_longitudinal=os.path.join(BASE_DIR,
                                                 'mask_longitudinal_petmr'
                                                 '.nii.gz'),
                 fmri_longitudinal=os.path.join(BASE_DIR,
                                                'mask_longitudinal_fmri'
                                                '.nii.gz'))


def load_atlas(atlas_name):
    """Retruns selected atlas path
        atlas_names values are : msdl, harvard_oxford, juelich, mayo ...
    """
    CACHE_DIR = _get_cache_base_dir()
    if atlas_name == 'msdl':
        from nilearn.datasets import load_atlas_msdl
        atlas = load_atlas_msdl()['maps']
    elif atlas_name == 'harvard_oxford':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'HarvardOxford-cortl-maxprob-thr0-2mm.nii.gz')
    elif atlas_name == 'juelich':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'Juelich-maxprob-thr0-2mm.nii.gz')
    elif atlas_name == 'julich':
            atlas = os.path.join(CACHE_DIR, 'atlas',
                                 'Juelich-prob-2mm.nii.gz')
    elif atlas_name == 'mayo':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'atlas_68_rois.nii.gz')
    elif atlas_name == 'canica':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'atlas_canica_61_rois.nii.gz')
    elif atlas_name == 'canica141':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'atlas_canica_141_rois.nii.gz')
    elif atlas_name == 'tvmsdl':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'tvmsdl_abide.nii.gz')
    elif atlas_name == 'kmeans':
        atlas = os.path.join(CACHE_DIR, 'atlas',
                             'atlas_kmeans.nii.gz')
    else:
        raise OSError('Atlas not found !')

    return atlas


def intersect_datasets(dataset1, dataset2, intersect_on='exam_codes'):
    """Returns the intersection of two dataset Bunches.
        The output is a dataset (Bunch).
        The intersection is on patient id and visit code or date
    """
    if intersect_on not in ['exam_codes', 'exam_dates']:
        raise ValueError('intersect_on should be either '
                         'exam_codes or exam_dates')
        return -1

    if 'subjects' not in dataset1.keys() or 'subjects' not in dataset2.keys():
        raise ValueError('Cannot intersect, Subject ID not found !')
        return -1

    if (intersect_on not in dataset1.keys() or
       intersect_on not in dataset2.keys()):
        raise ValueError('Cannot intersect,' + intersect_on + ' not found !')
        return -1
    return 0


def extract_baseline_dataset(dataset):
    """Returns baseline bunch of a dataset
    """
    # equivalent keys are : 'sc', 'bl', 'scmri'
    idx = np.hstack((np.where(dataset.exam_codes2 == 'sc'),
                     np.where(dataset.exam_codes2 == 'bl'),
                     np.where(dataset.exam_codes2 == 'scmri'))).ravel()

    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
        dataset[k] = dataset[k][idx]

    return dataset


def extract_unique_dataset(dataset):
    """Returns unique bunch of a dataset
    """
    _, unique_idx = np.unique(dataset.subjects, return_index=True)
    for k in dataset.keys():
        dataset[k] = np.array(dataset[k])
        dataset[k] = dataset[k][unique_idx]
    return dataset


def get_demographics(subjects, exam_dates=None):
    """Returns demographic informations (dob, gender)
    """
    BASE_DIR = _get_data_base_dir('ADNI_csv')
    demog = pd.read_csv(os.path.join(BASE_DIR, 'PTDEMOG.csv'))
    roster = pd.read_csv(os.path.join(BASE_DIR, 'ROSTER.csv'))
    mmse = pd.read_csv(os.path.join(BASE_DIR, 'MMSE.csv'))
    cdr = pd.read_csv(os.path.join(BASE_DIR, 'CDR.csv'))
    gdscale = pd.read_csv(os.path.join(BASE_DIR, 'GDSCALE.csv'))
    faq = pd.read_csv(os.path.join(BASE_DIR, 'FAQ.csv'))
    npiq = pd.read_csv(os.path.join(BASE_DIR, 'NPIQ.csv'))
    adas1 = pd.read_csv(os.path.join(BASE_DIR, 'ADASSCORES.csv'))
    adas2 = pd.read_csv(os.path.join(BASE_DIR, 'ADAS_ADNIGO2.csv'))
    nss = pd.read_csv(os.path.join(BASE_DIR, 'UWNPSYCHSUM_01_12_16.csv'))
    neurobat = pd.read_csv(os.path.join(BASE_DIR, 'NEUROBAT.csv'))

    # caching dataframe extraction functions
    CACHE_DIR = _get_cache_base_dir()
    cache_dir = os.path.join(CACHE_DIR, 'joblib', 'load_data_cache')
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    memory = Memory(cachedir=cache_dir, verbose=0)

    def _get_ridsdemo(subjects):
        return map(lambda s: _ptid_to_rid(s, roster), subjects)
    rids = np.array(memory.cache(_get_ridsdemo)(subjects))

    def _get_dobdemo(rids):
        return map(lambda r: _get_dob(r, demog), rids)
    dobs = np.array(memory.cache(_get_dobdemo)(rids))
    if exam_dates is not None:
        # compute age
        age = [np.round(abs(e - d).days/365., decimals=2)
               for e, d in zip(exam_dates, dobs)]

    def _get_genderdemo(rids):
        return map(lambda r: _get_gender(r, demog), rids)
    genders = np.array(memory.cache(_get_genderdemo)(rids)).astype(int)

    def _get_mmsedemo(rids):
        return map(lambda r: _get_mmse(r, mmse), rids)
    mmses = np.array(memory.cache(_get_mmsedemo)(rids))

    def _get_cdrdemo(rids):
        return map(lambda r: _get_cdr(r, cdr), rids)
    cdrs = np.array(memory.cache(_get_cdrdemo)(rids))

    def _getgdscaledemo(rids):
        return map(lambda r: _get_gdscale(r, gdscale), rids)
    gds = np.array(memory.cache(_getgdscaledemo)(rids))

    def _getfaqdemo(rids):
        return map(lambda r: _get_faq(r, faq), rids)
    faqs = np.array(memory.cache(_getfaqdemo)(rids))

    def _getnpiqdemo(rids):
        return map(lambda r: _get_npiq(r, npiq), rids)
    npiqs = np.array(memory.cache(_getnpiqdemo)(rids))

    def _getadasdemo(rids):
        return map(lambda r: _get_adas(r, adas1, adas2), rids)
    adas = np.array(memory.cache(_getadasdemo)(rids))

    def _getnssdemo(rids):
        return (map(lambda r: _get_nss(r, nss, mode=1), rids),
                map(lambda r: _get_nss(r, nss, mode=2), rids))
    nss1, nss2 = memory.cache(_getnssdemo)(rids)
    nss1, nss2 = np.array(nss1), np.array(nss2)

    def _getneurobatdemo(rids):
        return (map(lambda r: _get_neurobat(r, neurobat, mode=1), rids),
                map(lambda r: _get_neurobat(r, neurobat, mode=2), rids))
    nb1, nb2 = memory.cache(_getneurobatdemo)(rids)
    nb1, nb2 = np.array(nb1), np.array(nb2)

    if exam_dates is not None:
        return Bunch(dob=dobs, gender=genders, mmse=mmses,
                     nss1=nss1, nss2=nss2,
                     cdr=cdrs, gdscale=gds, faq=faqs, npiq=npiqs, adas=adas,
                     ldel=nb1, limm=nb2, age=age)
    else:
        return Bunch(dob=dobs, gender=genders, mmse=mmses,
                     nss1=nss1, nss2=nss2, cdr=cdrs, gdscale=gds,
                     faq=faqs, npiq=npiqs, adas=adas,
                     ldel=nb1, limm=nb2)


def load_longitudinal_dataset(modality='pet', nb_imgs_min=3, nb_imgs_max=5):
    """ Extract longitudinal images
    """

    if modality == 'pet':
        dataset = load_adni_longitudinal_fdg_pet()
        img_key = 'pet'
    elif modality == 'av45':
        dataset = load_adni_longitudinal_av45_pet()
        img_key = 'pet'
    elif modality == 'fmri':
        dataset = load_adni_longitudinal_rs_fmri_DARTEL()
        img_key = 'func'
    elif modality == 'csf':
        dataset = load_adni_longitudinal_csf_biomarker()
        # transform data as list of arrays
        dataset['csf'] = np.array(map(lambda C: '_'.join([str(c) for c in C]),
                                  dataset.csf))
        img_key = 'csf'
    elif modality == 'hippo':
        dataset = load_adni_longitudinal_hippocampus_volume()
        # transform data as list of arrays
        dataset['hipp'] = np.array(map(lambda H: '_'.join([str(h) for h in H]),
                                   dataset.hipp))
        img_key = 'hipp'
    else:
        raise ValueError('%s not found !' % modality)

    df = pd.DataFrame(data=dataset)
    grouped = df.groupby('subjects').groups

    df_count = df.groupby('subjects')[img_key].count()
    df_count = df_count[df_count >= nb_imgs_min]
    df_count = df_count[df_count <= nb_imgs_max]

    # n_images per subject
    # img_per_subject = df_count.values
    # unique subjects with multiple images
    subjects = df_count.keys().values
    subj = np.array([dataset.subjects[grouped[s]] for s in subjects])
    # diagnosis of the subjects
    dx_group = np.hstack([dataset.dx_group[grouped[s][0]] for s in subjects])
    dx_all = np.array([dataset.dx_group[grouped[s]] for s in subjects])
    # all images of the subjects
    imgs = np.array([dataset[img_key][grouped[s]] for s in subjects])
    imgs_baseline = np.array([dataset[img_key][grouped[s][0]]
                             for s in subjects])
    # csf and hipp cases
    if img_key in ['hipp', 'csf']:
        imgs = np.array(map(lambda img: [i.split('_') for i in img], imgs))
        imgs_baseline = [img.split('_') for img in imgs_baseline]
        imgs_baseline = np.array(map(lambda imgs: [np.float(i) for i in imgs],
                                     imgs_baseline))

    # acquisition and exam dates / codes of the subjects
    if 'exam_dates' in dataset.keys():
        exams = np.hstack([dataset.exam_dates[grouped[s][0]]
                          for s in subjects])
        exams_all = np.array([dataset.exam_dates[grouped[s]]
                             for s in subjects])

    if 'exam_codes' in dataset.keys():
        exams = np.hstack([dataset.exam_codes[grouped[s][0]]
                          for s in subjects])
        exams_all = np.array([dataset.exam_codes[grouped[s]]
                             for s in subjects])

    # age
    if modality in ['pet', 'av45']:
        ages_baseline = np.hstack([dataset.ages[grouped[s][0]]
                                   for s in subjects])
        ages = np.array([dataset.ages[grouped[s]] for s in subjects])
        return Bunch(imgs=imgs, imgs_baseline=imgs_baseline,
                     dx_group=dx_all, dx_group_baseline=dx_group,
                     subjects=subj, subjects_baseline=subjects,
                     ages=ages, ages_baseline=ages_baseline,
                     exams=exams_all, exams_baseline=exams,)
    else:
        return Bunch(imgs=imgs, imgs_baseline=imgs_baseline,
                     dx_group=dx_all, dx_group_baseline=dx_group,
                     subjects=subj, subjects_baseline=subjects,
                     exams=exams_all, exams_baseline=exams,)


def get_scores_adnidod(subjects):
    # data files
    BASE_DIR = _get_data_base_dir('ADNIDOD_csv')

    # meta-data
    demog = pd.read_csv(os.path.join(BASE_DIR, 'PTDEMOG.csv'))
    mmse = pd.read_csv(os.path.join(BASE_DIR, 'MMSE.csv'))
    cdr = pd.read_csv(os.path.join(BASE_DIR, 'CDR.csv'))
    gdscale = pd.read_csv(os.path.join(BASE_DIR, 'GDSCALE.csv'))
    faq = pd.read_csv(os.path.join(BASE_DIR, 'FAQ.csv'))
    npiq = pd.read_csv(os.path.join(BASE_DIR, 'NPI.csv'))
    adas = pd.read_csv(os.path.join(BASE_DIR, 'ADAS.csv'))
    neurobat = pd.read_csv(os.path.join(BASE_DIR, 'NEUROBAT.csv'))

    def get_score(subj_id, score, score_file, ptid='SCRNO'):
        m = score_file[score_file[ptid] == int(subj_id)][score].dropna().values
        if len(m) > 0:
            m[m < 0] = 0
            return np.median(m)
        else:
            return 0.

    df = {'subjects': subjects}
    keys = ['mmse', 'cdr', 'gdscale', 'faq', 'npiq', 'adas1', 'adas2',
            'ldel', 'limm', 'age']
    scores = ['MMSCORE', 'CDGLOBAL', 'GDTOTAL', 'FAQTOTAL', 'NPITOTAL',
              'TOTSCORE', 'TOTAL13', 'LDELTOTAL', 'LIMMTOTAL', 'PTAGE']
    score_files = [mmse, cdr, gdscale, faq, npiq, adas, adas, neurobat,
                   neurobat, demog]
    for k, s, sf in zip(keys, scores, score_files):
        sc = [get_score(subj, s, sf) for subj in subjects]
        df[k] = np.array(sc)
    return df


def load_adnidod_rs_fmri():
    """loader for adnidod rs fmri
    """
    subjects, subject_paths, description = _get_subjects_and_description(
        base_dir='ADNIDOD_rs_fmri', prefix='0*')
    subjects = np.array(subjects)
    # get func files
    func_files = map(lambda x: _glob_subject_img(x, suffix='func/' + 'wr*',
                                                 first_img=True),
                     subject_paths)
    func_files = np.array(func_files)
    scores = get_scores_adnidod(subjects)
    return Bunch(func=func_files,
                 subjects=subjects,
                 npiq=scores['npiq'],
                 mmse=scores['mmse'],
                 ldel=scores['ldel'],
                 age=scores['age'],
                 faq=scores['faq'],
                 cdr=scores['cdr'],
                 limm=scores['limm'],
                 adas1=scores['adas1'],
                 adas2=scores['adas2'],
                 gdscale=scores['gdscale'],)


def load_adnidod_av45_pet():
    """loader for adnidod rs fmri
    """
    subjects, subject_paths, description = _get_subjects_and_description(
        base_dir='ADNIDOD_av45_pet', prefix='0*')
    subjects = np.array(subjects)
    # get func files
    func_files = map(lambda x: _glob_subject_img(x, suffix='pet/' + 'wr*.nii',
                                                 first_img=True),
                     subject_paths)
    pet = np.array(func_files)
    scores = get_scores_adnidod(subjects)
    return Bunch(pet=pet,
                 subjects=subjects,
                 npiq=scores['npiq'],
                 mmse=scores['mmse'],
                 ldel=scores['ldel'],
                 age=scores['age'],
                 faq=scores['faq'],
                 cdr=scores['cdr'],
                 limm=scores['limm'],
                 adas1=scores['adas1'],
                 adas2=scores['adas2'],
                 gdscale=scores['gdscale'],)
