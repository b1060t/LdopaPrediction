%spmPath = '~/matlab_plugins/spm12';
spm('defaults','fmri');
spm_jobman('initcfg');
smooth_job = struct;

data = jsondecode(fileread('pat_train.json'));
nii = {data.CAT12_SGM};
nii = reshape(nii,length(nii),1);
nii = cellstr(nii);
age = [data.AGE];
sex = [data.SEX];
cat = [data.CAT];

spm_jobman('run',smooth_job.matlabbatch);