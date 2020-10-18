rng(1); %for reproducibility, anchoring randomization


load('data.mat')

dataFeatures = data.features;

%c = cvpartition(size(data.labels,1),'LeaveOut');
c = cvpartition(size(data.labels,1),'KFold',5);



test_labels_vector = []; % store ground truth in test order
predicted_labels_vector = [];  %vector to collect test results (labels assigned by SVM)
%accuracy = zeros(size(data.labels,1),1); %store accuracy from each test
%decision_score = zeros(size(data.labels,1),1); %vector of decision values (indep.of treshold)

for i = 1:c.NumTestSets % (LOO) run SVM 84 times

    trainIndex = c.training(i);
    testIndex = c.test(i);
    train_labels = data.labels(trainIndex);
    train_data = dataFeatures(trainIndex,:);
    test_labels = data.labels(testIndex);
    test_data = dataFeatures(testIndex,:);

   % the classifier 
    model = svmtrain(train_labels,train_data,'-t 0'); % training the classifier using the trainign data
    
    %[predicted_label, accuracy, decision_values] = svmpredict(test_labels,test_data,model); % testing the classfier on the left out data (hidden/test data)

    
    [predicted_labels, accuracy, decision_values] = svmpredict(test_labels,test_data,model); % testing the classfier on the left out data (hidden/test data)
       
    predicted_labels_vector = [predicted_labels_vector ; predicted_labels]; % concatenate predicted labels
    test_labels_vector = [test_labels_vector ; test_labels];
         
end


%% Classifier performance (classification accuracy)
%%


%cp = classperf(groundTruth,classifierOutput)
%cp = classperf(test_labels_vector,predicted_labels)


%% Confusion matrix
%%
CM = confusionmat(test_labels_vector,predicted_labels_vector); %returns the confusion matrix CM determined by the known and predicted groups in group and grouphat, respectively
% CM(i,j) is a count of observations known to be in group i but predicted to be in group j.

True_Negative = CM(1,1);
True_Positive = CM(2,2);
False_Negative = CM(2,1);
False_Positive = CM(1,2);
Accuracy = (True_Positive + True_Negative)/(size(data.labels,1)) * 100;
Sensitivity = (True_Positive)/(True_Positive + False_Negative) * 100;
Specificity = (True_Negative)/(True_Negative + False_Positive) * 100;

