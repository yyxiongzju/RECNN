clc;

load('trainwarp.mat');
load('testwarp.mat');
load('leaveout.mat');

saved_model_path = sprintf('%s', 'WARP_pretrained_model.mat');
load(saved_model_path);

max_iterations = 10;
clusters = 88;
uids = 1:clusters;
featureTrainMat = train_features;
featureTestMat = test_features;
trainDir = train_target;
testDir = test_target;

Z_train = featureTrainMat;
Z_test = featureTestMat;

stest_target = test_target;
stest_preds = test_preds;

cnn3d_loss = immse(stest_target, stest_preds);

[phi_fix, phi_lme] = em_lmecnn_evaluate(lf_phi, blfs_phi, b_hat_phi, featureTestMat, Z_test, featureTrainMat, trainIDs);
[theta_fix, theta_lme] = em_lmecnn_evaluate(lf_theta, blfs_theta, b_hat_theta, featureTestMat, Z_test, featureTrainMat, trainIDs);

pred_fix = [phi_fix, theta_fix];
pred_lme = [phi_lme, theta_lme];

recnn_fix_loss = immse(pred_fix, testDir);
recnn_lme_loss = immse(pred_lme, testDir);

exp_rmse_results = sprintf('%s:%f, %s:%f, %s:%f', 'RECNN Mixed RMSE', sqrt(recnn_lme_loss), 'RECNN Fix RMSE', sqrt(recnn_fix_loss), '3DCNN RMSE', sqrt(cnn3d_loss))