clc;

load('trainwarp.mat');
load('testwarp.mat');
load('leaveout.mat');

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

[phi_lf, phi_b_hat_history, phi_gll_history] = em_lmecnn(featureTrainMat, Z_train, uids, trainDir(:, 1), trainIDs, max_iterations);
blfs_phi = lme_train(phi_lf{max_iterations}, phi_b_hat_history{max_iterations}, featureTestMat, Z_test, featureTrainMat, trainIDs);
[theta_lf, theta_b_hat_history, theta_gll_history] = em_lmecnn(featureTrainMat, Z_train, uids, trainDir(:, 2), trainIDs, max_iterations);
blfs_theta = lme_train(theta_lf{max_iterations}, theta_b_hat_history{max_iterations}, featureTestMat, Z_test, featureTrainMat, trainIDs);

[min_phi_gll, min_phi_index] = min(phi_gll_history);
[min_theta_gll, min_theta_index] = min(theta_gll_history);

lf_phi = phi_lf{min_phi_index};
lf_theta = theta_lf{min_theta_index};
b_hat_phi = phi_b_hat_history{min_phi_index};
b_hat_theta = theta_b_hat_history{min_theta_index};

saved_model_path = sprintf('%s', 'WARP_pretrained_model.mat');
save(saved_model_path, 'lf_phi', 'lf_theta', 'b_hat_phi', 'b_hat_theta', 'blfs_phi', 'blfs_theta');
