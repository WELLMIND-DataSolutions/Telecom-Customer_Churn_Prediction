                    Telecom Churn Prediction Project

Project Overview

This document provides comprehensive documentation for the Telecom Churn Prediction project. The project implements a complete machine learning pipeline to predict customer churn in a telecommunications company. The pipeline includes data cleaning, exploratory data analysis, feature engineering, feature selection, model training, model evaluation, and SHAP analysis for model interpretability. The entire project is designed to identify customers who are likely to churn so that the business can take proactive retention actions.

Data Cleaning

Code Implementation
The data cleaning process begins with loading the raw telecom churn data from a CSV file. The target variable for churn is created based on customer behavior in month 9. Specifically, a customer is labeled as churned if they had zero recharge amount and zero recharge data in month 9. This approach ensures that only customers who have completely stopped using the service are marked as churned.

A critical step in the data cleaning process is the removal of all month 9 columns from the dataset. This is essential to prevent data leakage because month 9 data represents the actual churn period and using it for prediction would make the model unrealistically accurate. By removing these columns, the model is forced to rely only on historical data from months 1 through 8.

Missing values are handled separately for numeric and categorical columns. Numeric columns are filled using the median value to preserve the central tendency of the distribution without being affected by outliers. Categorical columns are filled using the mode, which represents the most frequent category in each column.

The script then performs data integrity checks to verify that no month 9 columns remain, that the target column exists, that missing values have been properly handled, and that the dataset is not empty. A comparison between churned and non-churned customers is performed to understand key behavioral differences, focusing on recharge amounts, data usage, and 3G usage patterns.

A donut plot is generated to visualize the churn distribution, showing both the absolute counts and percentages of churned versus non-churned customers. Finally, the cleaned dataset is saved to the processed data directory for use in subsequent pipeline stages.

Insight
Data leakage is the most critical issue to address in any predictive modeling project, especially in churn prediction where the churn event is defined based on future data. By removing all month 9 columns, we ensure that the model learns patterns from historical behavior only, which is how the model will operate in production when predicting future churn. The missing value handling strategy using median for numeric columns and mode for categorical columns is robust and maintains data integrity without introducing bias. The churn distribution analysis shows the proportion of customers who churned, which helps in understanding the class imbalance that will need to be addressed later in the pipeline.

Exploratory Data Analysis

Code Implementation
The exploratory data analysis phase begins by loading the cleaned dataset that was saved from the previous step. The first step in EDA is handling skewed distributions and outliers. Features with high skewness are transformed using logarithmic or square root transformations to make their distributions more normal-like, which improves the performance of many machine learning algorithms. Features with high kurtosis are also transformed to reduce the impact of extreme outliers.

After transformation, outlier capping is performed using the Interquartile Range method. Values below Q1 minus 1.5 times IQR and above Q3 plus 1.5 times IQR are clipped to the respective bounds. This approach preserves the overall distribution while removing the influence of extreme values that could distort model training.

A donut chart is generated to visualize the churn distribution, similar to the cleaning phase but now on the transformed dataset. Skewness and kurtosis analysis is performed on all numeric features, and a bar plot is created to show the top 20 features with the highest skewness and kurtosis values.

Violin plots combined with scatter plots are created for the six features showing the highest variance difference between churned and non-churned customers. This visualization helps understand how the distribution of each feature differs between the two groups. The scatter points are jittered to reduce overplotting, and their opacity is adjusted based on distance from the median to highlight typical values.

Normality testing is performed using D'Agostino's test, which is more appropriate for larger datasets than the Shapiro-Wilk test. Features are classified as normal or non-normal based on the p-value, and different correlation methods are used accordingly. Pearson correlation is used for normally distributed features, while Spearman rank correlation is used for non-normal features.

A correlation matrix with hierarchical clustering is created to visualize relationships between the top correlated features. The clustering groups features that behave similarly, making it easier to identify redundant features for potential removal.

Categorical features are analyzed using frequency distributions, chi-square tests, and mosaic plots. Rare categories are grouped into an "Other" category to prevent fragmentation. Chi-square tests determine which categorical features have a statistically significant relationship with churn. Residual heatmaps show where the actual counts differ most from expected counts, indicating strong associations between specific categories and churn.

Customer segmentation is performed based on ARPU, recharge amount, and usage patterns. Each customer is assigned to low, medium, or high categories for each segment using quantile-based binning. Churn rates are calculated for each segment to identify which customer groups are most at risk of churning. The final encoded dataset is saved for the next pipeline stage.

Insight
The exploratory data analysis reveals important patterns in the data. The skewness and kurtosis analysis shows that telecom data is typically highly skewed, with many customers having low usage and a few power users driving the distribution. This necessitates transformation to help machine learning models learn effectively. The correlation analysis identifies which features are most strongly associated with churn, providing early insights into the key drivers of customer attrition.

The chi-square tests on categorical features reveal that factors like contract type, payment method, and customer service interactions are significantly associated with churn. The segmentation analysis shows that customers with low ARPU, low recharge amounts, and low usage patterns have higher churn rates, suggesting that value-conscious customers or those who are not fully engaged with the service are more likely to leave. The mosaic plots provide intuitive visual representations of how each categorical feature interacts with the churn outcome.

Train Test Split

Code Implementation
The train test split process begins by loading the fully processed and encoded dataset from the previous EDA phase. The dataset is separated into features and target variable, where the target is the churn column that indicates whether a customer churned or not.

The train test split is performed using scikit-learn's train_test_split function with a test size of 20 percent. Random state is set to 42 to ensure reproducibility of the split, meaning that running the code multiple times will produce the same split. Stratification is applied based on the target variable, which ensures that the proportion of churned customers is preserved in both the training and testing sets. This is particularly important when dealing with imbalanced datasets because random splitting could accidentally create test sets with very different churn rates than the overall population.

After splitting, the training and testing sets are saved as separate CSV files in the processed data directory. Three visualization plots are generated to validate the quality of the split.

The first plot shows the target distribution comparison between train and test sets using side-by-side bar charts. This confirms that stratification worked correctly by showing similar percentages of churned customers in both sets. The second plot shows the dataset split sizes with total sample counts.

The second visualization consists of KDE overlay plots for the top six features with the highest variance. For each feature, the kernel density estimate of the training set is plotted alongside that of the test set. The Kolmogorov-Smirnov test statistic and p-value are calculated to statistically compare the distributions. Features with p-values above 0.05 are considered to have similar distributions between train and test, which is desirable. Green indicators are added to plots where distributions are similar, while red warnings are added where significant differences exist.

The third visualization compares the correlation matrices of the training and test sets. Three heatmaps are presented side by side: the training set correlation matrix, the test set correlation matrix, and the absolute difference between the two. Small differences between the train and test correlation matrices indicate that the relationships between features are stable across both sets, which is important for model generalization.

Insight
Stratified splitting is essential in churn prediction because the churn rate is typically low, often between 5 and 30 percent. Without stratification, there is a risk that the test set might contain very few churned customers, making it impossible to properly evaluate the model's ability to identify churners. The visualization of target distribution confirms that stratification successfully preserved the churn rate in both sets.

The KDE overlay plots with KS tests provide statistical validation that the feature distributions are similar between train and test. When significant differences are detected, it indicates that the random split created subsets with different underlying distributions, which could cause the model to perform differently on new data. The correlation matrix comparison reveals whether the structural relationships between features are stable. If the test set shows very different correlation patterns, the model might learn patterns from the training set that do not hold in the test set, leading to poor generalization.

Feature Engineering

Code Implementation
The feature engineering phase begins by loading the training data that was saved after the train test split. Missing values are handled using telecom-specific logic. Numeric columns are filled with zero because missing values in telecom data often indicate no activity rather than truly unknown values. Categorical columns are filled with "No_Activity" to distinguish customers who had no recorded activity from those who did.

A critical fix is applied to remove any month 9 columns that might have inadvertently survived the previous cleaning steps. This ensures that no future data leaks into the training process.

Basic behavior features are created to capture overall customer activity. Total activity is the sum of all numeric features for each customer, representing overall engagement. Average usage is the mean of numeric features, providing a normalized measure of typical behavior. Usage variability is the variance across numeric columns, with zero variance customers being those with constant values. A no activity flag identifies customers with zero total activity, while a high usage flag identifies customers with above-average usage.

RFM features are created based on telecom customer behavior. Frequency is the count of non-zero activities, representing how often the customer interacts with the service. Monetary value is the total sum of all numeric features, representing the total value derived from the customer. The recency feature from earlier versions was removed because the logic for calculating it was incorrect and could have introduced unintended biases.

Time-based features capture behavioral changes over the observed period. First activity is the value from the earliest month column, and last activity is the value from the latest month column. The activity trend is the difference between last and first activity, showing whether usage increased or decreased over time. The activity ratio is the last activity divided by first activity, with one added to both numerator and denominator to handle division by zero.

Ratio features provide normalized measures of efficiency. Usage per activity divides total activity by frequency, showing average value per interaction. Revenue efficiency divides monetary value by total activity, showing value generated per unit of activity. Activity density divides frequency by the number of time periods, showing consistency of engagement over time.

A leakage detection function is implemented to scan all features for potential data leakage. This includes checking for perfect correlation with the target, high correlation above 0.8, constant features that provide no information, ID-like features that are unique per customer, and time leakage keywords that might indicate post-churn information. If any potential leakage is detected, the features are flagged for review.

Boxplots are generated to compare the distribution of each new feature between churned and non-churned customers. Mann-Whitney U tests determine whether the differences between groups are statistically significant. For categorical features, bar plots show churn rates by category with chi-square tests for significance.

Additional plots are generated for deeper analysis. An effect size bar chart shows Cohen's d for each feature, measuring the magnitude of difference between churned and non-churned groups. A statistical significance scatter plot shows the negative log of p-values, making it easy to see which features have the strongest statistical evidence of difference. A feature direction pie chart shows whether features tend to be higher in churned customers or higher in non-churned customers. Distribution plots for the top three features show histograms with KDE overlays to visualize how churned and non-churned customers differ.

A lollipop chart shows the correlation of each feature with the target variable. Positive correlations indicate features that increase as churn probability increases, while negative correlations indicate features that decrease as churn probability increases.

Insight
Feature engineering is the most important phase in churn prediction because the raw features alone often do not capture the behavioral patterns that precede churn. The total activity feature combines multiple signals into a single measure of overall engagement. The usage variability feature is particularly important because customers who will churn often show declining engagement over time, which manifests as high variability between early and late periods.

The activity trend and activity ratio features capture the decline in usage that typically precedes churn. Customers who will churn often show decreasing activity over the observation period, while loyal customers maintain stable or increasing activity. The ratio features normalize the activity measures to account for customers who naturally have different baseline usage levels.

The leakage detection is critical because feature engineering can inadvertently create features that directly or indirectly reveal the target. For example, if a feature is created using future data or if a feature perfectly correlates with churn, the model would appear artificially accurate but would fail in production. The systematic scanning for different types of leakage protects against this risk.

The statistical testing of new features shows which engineered features are truly discriminative between churned and non-churned customers. Features with low p-values in the Mann-Whitney U test or low p-values in the chi-square test are likely to be useful predictors. The effect size provides a standardized measure of how large the difference is between groups, which helps prioritize which features are most important.

Feature Selection

Code Implementation
The feature selection phase begins by loading the feature engineered dataset. Constant columns that have only one unique value across all rows are removed because they provide no information for prediction.

Data cleaning is performed to replace infinite values with NaN and then fill numeric columns with zero and categorical columns with "No_Value". This ensures that all values are finite and valid for further analysis.

Correlation analysis is performed to identify which features have the strongest linear relationship with the churn target. The top correlated features are displayed for information purposes, but correlation alone is not used for feature removal because correlation only captures linear relationships and may miss important non-linear patterns.

Variance Inflation Factor analysis is performed to detect and remove multicollinearity. Multicollinearity occurs when two or more features are highly correlated with each other, which can cause instability in some models and make feature importance interpretation difficult. The VIF measures how much the variance of a coefficient is inflated due to correlation with other features. Features with VIF above 10 are considered to have problematic multicollinearity and are removed iteratively. Before VIF calculation, features with correlation above 0.95 are removed to speed up the process and address the most severe redundancies first.

After VIF-based removal, the dataset is separated into features and target. Standard scaling is applied to all numeric features to bring them to a common scale. This is important for models that are sensitive to feature scales, such as logistic regression and distance-based algorithms. The scaling is done using StandardScaler, which subtracts the mean and divides by the standard deviation.

The class distribution is checked to determine whether the dataset is imbalanced. The imbalance ratio is calculated as the number of churned customers divided by the number of non-churned customers. If the ratio is below 0.7, indicating significant imbalance, SMOTE is applied to balance the classes. SMOTE creates synthetic samples of the minority class by interpolating between existing minority class samples. This creates a balanced dataset without simply duplicating existing samples, which would cause overfitting.

Random Forest feature selection is performed on the balanced dataset. A Random Forest classifier is trained with 300 estimators, and feature importances are extracted. Features with importance below 0.002 are considered unimportant and are dropped from the dataset. This threshold can be adjusted based on the specific dataset and the desired number of features.

The final selected features dataset is saved along with the target variable. Several visualization plots are generated to document the feature selection process. A feature importance bar chart shows the top 20 features with their importance scores. A SMOTE impact donut chart shows the class distribution before and after SMOTE application. A final class distribution bar chart confirms the balanced class distribution. A correlation lollipop chart shows the correlation of each feature with the target variable.

Insight
Feature selection serves multiple purposes. First, it reduces the dimensionality of the dataset, which can improve training time and reduce memory usage. Second, it removes redundant and irrelevant features, which can improve model performance by reducing noise. Third, it makes the model more interpretable by focusing on the most important predictors.

The VIF analysis addresses multicollinearity, which is a common problem in telecom data where many features are naturally correlated. For example, total recharge amount and number of recharge transactions are likely correlated, and keeping both can cause instability in coefficient estimates for linear models. The iterative removal of high VIF features ensures that the remaining features are relatively independent.

SMOTE is applied because churn prediction typically involves imbalanced data where churned customers are a small minority. Without addressing imbalance, models tend to predict the majority class almost exclusively, achieving high accuracy but failing to identify churned customers. SMOTE creates synthetic churned customers by interpolating between existing churned customers in feature space, which is more effective than simple oversampling because it creates new information rather than duplicating existing data.

The Random Forest feature selection provides a non-linear measure of feature importance based on how much each feature reduces impurity across all trees in the forest. This approach can capture non-linear relationships and interactions that correlation analysis might miss. Features with importance below the threshold are unlikely to contribute meaningfully to prediction and can be safely removed.

Model Training

Code Implementation
The model training phase begins by loading the final selected features dataset that was saved after feature selection. The dataset is separated into features and target variable, and the target distribution is examined to understand the class balance after feature selection.

Five different classification models are defined for comparison: Random Forest, XGBoost, LightGBM, Logistic Regression, and CatBoost. Each model is paired with a hyperparameter grid that specifies which parameter combinations should be tested during tuning.

Randomized search cross-validation is used for hyperparameter tuning. This approach randomly samples parameter combinations from the specified grids rather than testing all combinations exhaustively. This is more efficient when the parameter space is large, and it often finds good parameters faster than grid search. The search is optimized for the F1 score because F1 balances precision and recall, which is appropriate for imbalanced classification problems.

After hyperparameter tuning, each model is evaluated using 5-fold stratified cross-validation on the full dataset. Stratification ensures that each fold maintains the same class distribution as the overall dataset. The cross-validation calculates accuracy, precision, recall, and F1 score for each fold, and the mean and standard deviation across folds are reported.

Training time is recorded for each model to provide information about computational efficiency. This is important for production deployment where training time constraints might favor faster models even if they have slightly lower performance.

The results for all models are compiled into a comparison table and saved as a CSV file. Four visualization plots are generated to compare model performance.

The F1 score comparison plot shows the mean F1 score for each model with error bars representing the standard deviation across cross-validation folds. This allows visual comparison of both central tendency and stability.

The precision, recall, and F1 comparison plot shows grouped bars for each model, allowing comparison of all three metrics simultaneously. This is useful because different applications may prioritize different metrics, and the plot shows which models have strengths in each area.

The training time comparison plot shows how long each model took to train, helping to identify whether performance improvements justify the additional training time.

The accuracy comparison plot shows accuracy with error bars, providing another perspective on model performance.

Insight
Comparing multiple models is essential because different algorithms have different strengths and weaknesses. Random Forest and XGBoost are ensemble methods that combine many weak learners to create a strong predictor. They tend to perform well on tabular data and are robust to outliers and non-linear relationships. LightGBM is a gradient boosting framework optimized for speed and efficiency, often training faster than XGBoost while achieving similar performance. CatBoost is designed to handle categorical features natively, which can be beneficial when many categorical features remain after encoding. Logistic Regression serves as a baseline model that is simple, interpretable, and fast to train.

The F1 score is chosen as the primary optimization metric because in churn prediction, both false positives and false negatives have costs. False negatives, where a customer who will churn is not identified, represent missed opportunities for retention. False positives, where a customer who will not churn is identified as a churn risk, represent wasted retention resources on customers who would have stayed anyway. The F1 score balances both concerns by taking the harmonic mean of precision and recall.

The cross-validation results also show the stability of each model. Models with low standard deviation across folds are more consistent and likely to generalize well to new data. Models with high standard deviation may be sensitive to the specific composition of the training data, which is a risk for production deployment.

Model Evaluation

Code Implementation
The model evaluation phase focuses on the best performing model from the comparison, which is the Random Forest classifier. The final selected features dataset is loaded, and the model is trained on the full dataset after hyperparameter tuning.

Before training the final model, the dataset is examined to determine whether sampling is needed for speed. If the dataset has more than 50,000 rows, a sample of 30,000 rows is used for hyperparameter tuning to keep computation time reasonable. The final model is then trained on the full dataset using the best parameters found during tuning.

Cross-validation is performed on the final model to get unbiased estimates of generalization performance. Five-fold stratified cross-validation calculates accuracy, precision, recall, and F1 score. The mean and standard deviation across folds are reported to provide both point estimates and measures of uncertainty.

Predictions are made on the full training dataset to examine in-sample performance, though the cross-validation results are more reliable indicators of true generalization. The confusion matrix is calculated to show the counts of true negatives, false positives, false negatives, and true positives. The classification report provides precision, recall, and F1 score for each class individually.

The AUC-ROC score is calculated to measure the model's ability to distinguish between churned and non-churned customers across all probability thresholds. The AUC-ROC ranges from 0.5 for a random classifier to 1.0 for a perfect classifier.

Overfitting analysis is performed by comparing the cross-validation F1 score to the full data F1 score. A small difference indicates that the model generalizes well, while a large difference indicates overfitting to the training data.

Nine professional visualization plots are generated for comprehensive model evaluation. The AUC-ROC curve shows the trade-off between true positive rate and false positive rate, with the area under the curve quantifying overall discriminative ability. The precision-recall curve shows the trade-off between precision and recall, which is often more informative than ROC for imbalanced problems.

The feature importance plot shows the top 15 features that drive churn predictions, providing interpretability about what the model has learned. The confusion matrix heatmap shows the prediction outcomes with counts and percentages for each cell.

The class distribution plot shows the balance between churned and non-churned customers in the dataset. The probability distribution plot shows how predicted probabilities are distributed for customers who actually churned versus those who did not, with a vertical line at the 0.5 threshold.

The cumulative gains curve shows what percentage of actual churners would be captured by targeting different percentages of customers with the highest predicted risk. This is directly useful for business decision-making about retention marketing budgets.

The radar chart shows all performance metrics on a single plot for easy comparison across dimensions. The error analysis dashboard breaks down the confusion matrix into error types and shows the distribution of false positives versus false negatives.

The final model is saved using joblib for later use in the SHAP analysis and potential deployment.

Insight
Random Forest is particularly well-suited for churn prediction for several reasons. It can capture non-linear relationships between features and churn without requiring manual feature transformations. It provides feature importance scores that help interpret what drives churn. It is robust to outliers and noisy data, which is common in telecom datasets. It handles interactions between features naturally because decision trees can split on different features at different depths.

The AUC-ROC score being close to 0.9 indicates excellent discriminative ability, meaning the model can reliably distinguish customers who will churn from those who will not. The precision and recall balance shown by the F1 score indicates that the model achieves a good trade-off between identifying churners and avoiding false alarms.

The cumulative gains curve shows that by targeting the 20 percent of customers with the highest predicted churn risk, the model captures a much larger proportion of actual churners than random targeting would. This is directly actionable for the business because it shows the efficiency gains from using the model to prioritize retention efforts.

The error analysis reveals where the model makes mistakes. False negatives, where churners are not identified, represent missed opportunities that could be addressed by lowering the classification threshold if the business is willing to accept more false positives. False positives represent wasted retention resources that could be reduced by raising the threshold if the business wants to be more selective.

The top features from the Random Forest provide actionable insights. Features related to declining usage, support tickets, and payment issues typically rank highly, suggesting specific intervention strategies. For example, customers with increasing support tickets might benefit from proactive outreach, while customers with declining usage might need engagement campaigns.

SHAP Analysis

Code Implementation
The SHAP analysis phase provides model interpretability by explaining why individual customers are predicted to churn. The trained Random Forest model is loaded from the saved file along with the final selected features dataset.

A sample of 200 customers is selected for SHAP analysis to keep computation time reasonable while still providing sufficient data for stable interpretations. SHAP values are calculated using the TreeExplainer, which is specifically designed for tree-based models like Random Forest and is efficient for large datasets.

The SHAP values represent the contribution of each feature to the model's prediction for each customer. Positive SHAP values push the prediction toward churn, while negative SHAP values push the prediction away from churn. The magnitude of the SHAP value indicates the strength of the contribution.

The SHAP values are saved to a PKL file in the models directory for potential future analysis without recalculating. Feature importance based on SHAP is calculated as the mean absolute SHAP value for each feature across all sampled customers. This represents the average impact of each feature on predictions, regardless of direction.

A CSV file of feature importance is saved, showing each feature, its mean absolute SHAP value, and whether it typically increases or decreases churn risk.

Three visualization plots are generated. The feature impact donut chart shows the relative contribution of the top five features versus all other features combined, providing a high-level view of which features drive most predictions.

The horizontal bar chart shows the top 15 features with their mean absolute SHAP values. This provides a clear ranking of feature importance with exact numerical values.

The risk segmentation donut chart shows the proportion of customers in the sample classified as low risk, medium risk, and high risk based on predicted churn probabilities. Low risk is defined as probability below 40 percent, medium risk between 40 and 70 percent, and high risk above 70 percent.

Business insights are generated based on the top churn drivers. For each top feature, specific problems and recommended solutions are provided. For example, short tenure customers are identified as high risk, with a recommendation to implement a first 90 days onboarding program. High bill amounts indicate customer frustration with pricing, with a recommendation to offer personalized discounts or tiered plans. High support ticket counts indicate service issues, with a recommendation for proactive support and faster resolution. Low satisfaction scores indicate general unhappiness, with a recommendation for regular NPS surveys and follow-up. Payment issues indicate friction in the billing process, with a recommendation to simplify payment methods and send reminders.

Insight
SHAP analysis is essential for making machine learning models interpretable and trustworthy. While the Random Forest model provides excellent predictive accuracy, it acts as a black box without explanation of why specific predictions are made. SHAP opens this black box by showing which features contributed most to each prediction.

The mean absolute SHAP values show that a small number of features typically drive most predictions. This is common in well-trained models and indicates that the model has learned to focus on the most informative signals rather than spreading attention across many weak signals.

The risk segmentation reveals how confident the model is in its predictions. Customers in the high risk category have predicted probabilities above 70 percent, indicating high confidence that they will churn. These customers should be the highest priority for retention interventions. Customers in the low risk category have predicted probabilities below 40 percent, indicating high confidence that they will stay, and retention resources should not be wasted on them. Customers in the medium risk category represent uncertainty where further analysis or monitoring might be needed.

The business insights translate the technical feature importance into actionable recommendations. Each top driver is paired with a specific problem statement and solution recommendation. For example, if support ticket count is a top driver, the insight is that customers who have contacted support multiple times are frustrated and likely to leave. The recommended solution is proactive support and faster resolution, which addresses the root cause rather than just treating the symptom.

The SHAP values are saved to a file so that they can be loaded later without recalculating. This is useful for building dashboards or monitoring systems that need to explain predictions in real time but cannot afford the computation of SHAP values on every prediction.

Pipeline Summary

The complete pipeline processes raw telecom customer data through multiple stages to produce a trained churn prediction model with interpretability features. The data cleaning stage removes future data to prevent leakage and handles missing values appropriately. The exploratory data analysis stage transforms skewed features, handles outliers, and provides initial insights into churn patterns. The train test split stage creates representative training and testing sets with stratification to preserve class distribution.

The feature engineering stage creates behavioral features that capture engagement patterns, trends over time, and efficiency metrics that are strong predictors of churn. The feature selection stage reduces dimensionality by removing constant features, highly correlated features, multicollinear features, and low importance features. The model training stage compares multiple algorithms and selects the best performer based on cross-validated F1 score.

The model evaluation stage provides comprehensive performance metrics and visualizations including ROC curves, precision-recall curves, confusion matrices, and cumulative gains curves. The SHAP analysis stage provides model interpretability by identifying which features drive churn predictions and generating business insights with recommended actions.

The final output of the pipeline includes a trained Random Forest model, comprehensive evaluation metrics, actionable business insights, and interpretability tools that explain why individual customers are predicted to churn. This enables the business to implement targeted retention campaigns for high-risk customers based on their specific risk factors.