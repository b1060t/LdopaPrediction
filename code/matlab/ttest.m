%spmPath = '~/matlab_plugins/spm12';
spm('defaults','fmri');
spm_jobman('initcfg');
estimate_job = struct;

good = jsondecode(fileread('train_good.json'));
good_img = {good.CAT12_SGM};
good_img = reshape(good_img,length(good_img),1);
good_img = cellstr(good_img);
bad = jsondecode(fileread('train_bad.json'));
bad_img = {bad.CAT12_SGM};
bad_img = reshape(bad_img,length(bad_img),1);
bad_img = cellstr(bad_img);
sexlist = {good.SEX, bad.SEX};
sexlist = cell2mat(reshape(sexlist,length(sexlist),1));
agelist = {good.AGE, bad.AGE};
agelist = cell2mat(reshape(agelist,length(agelist),1));
durationlist = {good.DURATION, bad.DURATION};
durationlist = cell2mat(reshape(durationlist,length(durationlist),1));
tivlist = {good.TIV, bad.TIV};
tivlist = cell2mat(reshape(tivlist,length(tivlist),1));
leddlist = {good.LEDD, bad.LEDD};
leddlist = cell2mat(reshape(leddlist,length(leddlist),1));
gmlist = {good.GM_VOL, bad.GM_VOL};
gmlist = cell2mat(reshape(gmlist,length(gmlist),1));
wmlist = {good.WM_VOL, bad.WM_VOL};
wmlist = cell2mat(reshape(wmlist,length(wmlist),1));

matPath = "./spm";

estimate_job.matlabbatch{1}.spm.stats.factorial_design.dir = cellstr(matPath);
estimate_job.matlabbatch{1}.spm.stats.factorial_design.des.t2.scans1 = good_img;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.des.t2.scans2 = bad_img;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.des.t2.dept = 0;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.des.t2.variance = 1;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.des.t2.gmsca = 0;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.des.t2.ancova = 0;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(1).c = gmlist;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(1).cname = 'gm';
estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(1).iCFI = 1;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(1).iCC = 1;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(2).c = wmlist;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(2).cname = 'wm';
estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(2).iCFI = 1;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(2).iCC = 1;
%estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(3).c = durationlist;
%estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(3).cname = 'duration';
%estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(3).iCFI = 1;
%estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(3).iCC = 1;
%estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(4).c = tivlist;
%estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(4).cname = 'tiv';
%estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(4).iCFI = 1;
%estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(4).iCC = 1;
%estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(5).c = leddlist;
%estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(5).cname = 'ledd';
%estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(5).iCFI = 1;
%estimate_job.matlabbatch{1}.spm.stats.factorial_design.cov(5).iCC = 1;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
estimate_job.matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
estimate_job.matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
estimate_job.matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

spm_jobman('run',estimate_job.matlabbatch);

spmPath = "spm/SPM.mat";
spm('defaults','fmri');
spm_jobman('initcfg');
run_job = struct;
run_job.matlabbatch{1}.spm.stats.frmi_est.spmmat = cellstr(spmPath);
run_job.matlabbatch{1}.spm.stats.frmi_est.write_residuals = 0;
run_job.matlabbatch{1}.spm.stats.frmi_est.method.Classical = 1;

spm_jobman('run',run_job.matlabbatch);

spm('defaults','fmri');
spm_jobman('initcfg');
contrast_job = struct;
contrast_job.matlabbatch{1}.spm.stats.con.spmmat = cellstr(spmPath);
contrast_job.matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'good > bad';
contrast_job.matlabbatch{1}.spm.stats.con.consess{1}.tcon.convec = [1 -1 0 0];
contrast_job.matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
contrast_job.matlabbatch{1}.spm.stats.con.delete = 1;

spm_jobman('run',contrast_job.matlabbatch);