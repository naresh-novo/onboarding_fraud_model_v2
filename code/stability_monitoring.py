import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score


class novo_ml_metrics:
    """
        The types of metrics to be used and monitored in the novo models to evaluate the model performance.
    """

    def __init__(self) -> None:
        """Initialize the class"""
        pass


    def psi_equal_width(actual: np.ndarray, expected: np.ndarray, num_bins: int=10, labels: str=True, bins:str=None , plot: str=False):
        """
        Args:
            expected: an array of the expected values 
            actual: an array of the actual values
            num_bins: an integer between 1 and the size of the array
            labels: bins label to output
            plot: if True then returns a plot with binning distribution

        Returns:
            psi_score: a non-negative float value
        """
        eps = 1e-4

        # Prepare the bins
        if bins == None:
            min_val = min(actual)
            max_val = max(actual)
            bins = [min_val + (max_val - min_val)*(i)/num_bins for i in range(num_bins+1)]
            bins[0] = min(bins[0], min(expected)) - eps # Correct the lower boundary and make sure the lowest boundary is either from actual or expected
            bins[-1] = max(bins[-1], max(expected)) + eps # Correct the higher boundary and make sure the lowest boundary is either from actual or expected

        if labels:
            labels = range(1,num_bins+1)
    
        # Bucketize the actual population and count the sample inside each bucket
        bins_actual = pd.cut(actual, bins = bins, labels=labels)
        df_actual = pd.DataFrame({'base_sample': actual, 'bin': bins_actual})
        grp_actual = df_actual.groupby('bin').count()
        grp_actual['base_sample_percent'] = np.round(grp_actual['base_sample'] / actual.shape[0] ,3)
        
        # Bucketize the expected population and count the sample inside each bucket
        bins_expected = pd.cut(expected, bins = bins, labels = labels)
        df_expected = pd.DataFrame({'expected_sample': expected, 'bin': bins_expected})
        grp_expected = df_expected.groupby('bin').count()
        grp_expected['expected_sample_percent'] = np.round(grp_expected['expected_sample'] / expected.shape[0] ,3)
        
        # Compare the bins to calculate PSI
        psi_df = grp_actual.join(grp_expected, on = "bin", how = "inner")
        
        # Add a small value for when the percent is zero
        psi_df['base_sample_percent'] = psi_df['base_sample_percent'].apply(lambda x: eps if x == 0 else x)
        psi_df['expected_sample_percent'] = psi_df['expected_sample_percent'].apply(lambda x: eps if x == 0 else x)
        
        # Calculate the psi
        psi_df['psi'] = (psi_df['base_sample_percent'] - psi_df['expected_sample_percent']) * np.log(psi_df['base_sample_percent'] / psi_df['expected_sample_percent'])
        psi_score = np.round(sum(psi_df['psi']),3)

        if plot:
            psi_plot = psi_df.reset_index()
            psi_plot['base_sample_percent'] = psi_plot['base_sample_percent'] *100
            psi_plot['expected_sample_percent'] = psi_plot['expected_sample_percent'] *100            
            psi_plot = pd.melt(psi_plot[['bin', 'base_sample_percent', 'expected_sample_percent']], id_vars = 'bin', var_name = 'set', value_name = 'percent')
            
            plt.figure(figsize = (10,4))
            sns.barplot(x = 'bin', y = 'percent', hue = 'set', data = psi_plot, palette = 'Blues')
            plt.title(" distribution - equal_width bins")

        return psi_score, psi_df


    def psi_equal_frequency(actual: np.ndarray, expected: np.ndarray, num_bins: int = 10, labels: str = True, plot: str = False):
        """
        Returns a  PSI value using equal width/range method

        Args:
            expected: an array of the expected values 
            actual: an array of the actual values
            num_bins: an integer between 1 and the size of the array
            labels: bins label to output
            plot: if True then returns a plot with binning distribution

        Returns:
            psi_score: a non-negative float value
        """
        eps = 1e-4

        bins = pd.qcut(actual, q=num_bins, retbins=True, duplicates="drop")[1] # Create the bins based on the actual population
        num_bins = len(bins)-1 # recreating the no.of bins to make sure that the bins edges are distinct
        
        if labels:
            labels = range(1,num_bins+1)

        # Bucketize the actual population and count the sample inside each bucket
        bins[0] = min(bins[0], min(expected)) - eps # Correct the lower boundary and make sure the lowest boundary is either from actual or expected
        bins[-1] = max(bins[-1], max(expected)) + eps # Correct the higher boundary and make sure the lowest boundary is either from actual or expected

        bins_actual = pd.cut(actual, bins = bins, labels=labels)
        df_actual = pd.DataFrame({'base_sample': actual, 'bin': bins_actual})
        grp_actual = df_actual.groupby('bin').count()
        grp_actual['base_sample_percent'] = np.round(grp_actual['base_sample'] / actual.shape[0] ,3)
        
        # Bucketize the expected population and count the sample inside each bucket
        bins_expected = pd.cut(expected, bins = bins, labels = labels)
        df_expected = pd.DataFrame({'expected_sample': expected, 'bin': bins_expected})
        grp_expected = df_expected.groupby('bin').count()
        grp_expected['expected_sample_percent'] = np.round(grp_expected['expected_sample'] / expected.shape[0] ,3)
        
        # Compare the bins to calculate PSI
        psi_df = grp_actual.join(grp_expected, on = "bin", how = "inner")
        
        # Add a small value for when the percent is zero
        psi_df['base_sample_percent'] = psi_df['base_sample_percent'].apply(lambda x: eps if x == 0 else x)
        psi_df['expected_sample_percent'] = psi_df['expected_sample_percent'].apply(lambda x: eps if x == 0 else x)
        
        # Calculate the psi
        psi_df['psi'] = (psi_df['base_sample_percent'] - psi_df['expected_sample_percent']) * np.log(psi_df['base_sample_percent'] / psi_df['expected_sample_percent'])
        psi_score = np.round(sum(psi_df['psi']),3)

        if plot:
            psi_plot = psi_df.reset_index()
            psi_plot['base_sample_percent'] = psi_plot['base_sample_percent'] *100
            psi_plot['expected_sample_percent'] = psi_plot['expected_sample_percent'] *100            
            psi_plot = pd.melt(psi_plot[['bin', 'base_sample_percent', 'expected_sample_percent']], id_vars = 'bin', var_name = 'set', value_name = 'percent')
            
            plt.figure(figsize = (10,4))
            sns.barplot(x = 'bin', y = 'percent', hue = 'set', data = psi_plot, palette = 'Blues')
            plt.title(" distribution - equal_width bins")

        return psi_score, psi_df


    def psi_categorical(actual: np.ndarray, expected: np.ndarray, plot: str = False):
        """
        Returns a PSI value.

        Args:
            expected: an array of the expected values 
            actual: an array of the actual values
            plot: if True then returns a plot with binning distribution

        Returns:
            psi_score: a non-negative float value
        """

        eps = 1e-4

        df_actual = pd.DataFrame(actual, columns=['base'])
        df_expected = pd.DataFrame(expected, columns=['expected'])
        grp_actual = pd.DataFrame(df_actual['base'].value_counts(normalize=True)).reset_index(drop=False).rename(columns={'index':'category','base':'base_sample_percent'})
        grp_expected = pd.DataFrame(df_expected['expected'].value_counts(normalize=True)).reset_index(drop=False).rename(columns={'index':'category','expected':'expected_sample_percent'})
        grp_actual['base_sample_percent'] = np.round(grp_actual['base_sample_percent'],3)
        grp_expected['expected_sample_percent'] = np.round(grp_expected['expected_sample_percent'],3)

        # Compare the bins to calculate PSI
        psi_df = pd.merge(grp_actual,grp_expected, on='category', how='outer')
        
        # Add a small value for when the percent is zero
        psi_df['base_sample_percent'] = psi_df['base_sample_percent'].apply(lambda x: eps if x == 0 else x)
        psi_df['expected_sample_percent'] = psi_df['expected_sample_percent'].apply(lambda x: eps if x == 0 else x)
        
        # Calculate the psi
        psi_df['psi'] = (psi_df['base_sample_percent'] - psi_df['expected_sample_percent']) * np.log(psi_df['base_sample_percent'] / psi_df['expected_sample_percent'])
        psi_score = np.round(sum(psi_df['psi']),3)

        if plot:
            psi_plot = psi_df.reset_index()
            psi_plot['base_sample_percent'] = psi_plot['base_sample_percent'] *100
            psi_plot['expected_sample_percent'] = psi_plot['expected_sample_percent'] *100
            psi_plot = pd.melt(psi_plot[['category', 'base_sample_percent', 'expected_sample_percent']], id_vars = 'category', var_name = 'set', value_name = 'percent')
            
            plt.figure(figsize = (10,4))
            sns.barplot(x = 'category', y = 'percent', hue = 'set', data = psi_plot, palette = 'Blues')
            plt.title(" distribution - categorical bins")

        return psi_score, psi_df


    def bin_ks_score(target:np.ndarray, prob:np.ndarray, num_bins:int=10, labels:str=True, bins=None):
        """
        Returns a max of KS value of the bins.

        Args:
            target: true label array
            prob: probabilities array
            num_bins: number of bins with the equal frequency
            labels: True, None or labels list
            bins: list of the lower edges of the bins

        Returns:
            ks_score: a non-negative float value ranges from 0 to 100
            ks_score_bin: The bin number with the max KS score
            kstable: Dataframe with binwise details
        """
        target_col = 'target'
        prob_col = 'prob'
        num_bins = num_bins
        bins = bins
        data = pd.DataFrame(zip(target,prob), columns=[target_col, prob_col])

        if bins == None:
            bins = pd.qcut(data[prob_col], q=num_bins, retbins=True, duplicates="drop")[1] # Create the bins based on the population
        elif type(bins) == list and len(bins) >=2:
            bins = bins
        
        num_bins = len(bins)-1 # recreating the no.of bins to make sure that the bins edges are distinct
        
        if labels:
            labels = range(1,num_bins+1)
        
        data['bucket'] = pd.cut(data[prob_col], bins=bins, labels=labels)

        data['target0'] = 1 - data[target_col]
        grouped = data.groupby('bucket', as_index = False)

        # Extract all the events and non-events for each bin/bucket
        kstable = pd.DataFrame()
        kstable['bin'] =grouped.min()['bucket']
        kstable['min_prob'] = grouped.min()[prob_col]
        kstable['max_prob'] = grouped.max()[prob_col]
        kstable['events']   = grouped.sum()[target_col]
        kstable['nonevents'] = grouped.sum()['target0']
        kstable = kstable.sort_values(by="min_prob", ascending=False).reset_index(drop = True)
        kstable['event_rate'] = np.round(kstable.events / data[target_col].sum(),3)
        kstable['nonevent_rate'] = np.round(kstable.nonevents / data['target0'].sum(),3)
        kstable['cum_eventrate'] = np.round((kstable.events / data[target_col].sum()).cumsum(),3)
        kstable['cum_noneventrate'] = np.round((kstable.nonevents / data['target0'].sum()).cumsum(),3)
        kstable['volume'] = (kstable.events + kstable.nonevents)*100/(kstable.events.sum() + kstable.nonevents.sum())
        kstable['cum_vol_percent'] = np.round(kstable.volume.cumsum(),0)
        # Calculate the KS score at each bin
        kstable['KS'] = np.round((kstable['cum_eventrate']-kstable['cum_noneventrate']) * 100 ,3)
        kstable.index.rename('bin', inplace=True) # Set Index as bin column
        ks_score = max(kstable['KS']) # Get the max KS score
        ks_score_bin = kstable.index[kstable['KS']==max(kstable['KS'])][0] + 1  # Get index of max KS score. Also, added +1 to start bin_no. from 1.
        
        return ks_score, ks_score_bin, kstable


    def auc_gini_score(target:np.ndarray, prob:np.ndarray, average:str='macro'):
        """
        Returns AUC and GINI scores.

        Args:
            target: true label array
            prob: probabilities array
            average: {‘micro’, ‘macro’, ‘samples’, ‘weighted’} or None, default=’macro’

        Returns:
            A dict with the below values
            auc: a non-negative float value ranges from 0.5 to 1. If the value is less than 0.5 then
                 something wrong with the model performance.
            gini: a non-negative float value ranges from 0 to 1. If the value is negative then
                  something wrong with the model performance.
        """

        target_col = 'target'
        prob_col = 'prob'
        average = average
        data = pd.DataFrame(zip(target,prob), columns=[target_col, prob_col])

        metrics_dict = {}
        metrics_dict['auc'] = roc_auc_score(data[target_col], data[prob_col], average=average)
        metrics_dict['gini'] = 2*metrics_dict['auc'] - 1
        return metrics_dict


    def precision_recall_sensitivity_specificity(target:np.ndarray, prob:np.ndarray, average:str='binary', threshold:float=0.5, pos_label:int=1):
        """
        Returns precision, recall, specificity and sensitivity scores.

        Args:
            target: true label array
            prob: probabilities array
            average: {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} or None, default=’binary’
            threshold: A value from 0 to 1, and used to asign the predicted label. default=0.5
            pos_label: An int value of class 0 or 1, default=1

        Returns:
            A dict with the below values
            precision: a non-negative float value ranges from 0 to 1
            recall: a non-negative float value ranges from 0 to 1
            specificity: a non-negative float value ranges from 0 to 1
            sensitivity: a non-negative float value ranges from 0 to 1
        """
        target_col = 'target'
        prob_col = 'prob'
        data = pd.DataFrame(zip(target,prob), columns=[target_col, prob_col])

        pred_labels = (data[prob_col].values >= threshold).astype('int')
        true_labels = data[target_col].values
        
        metrics_dict = {}
        metrics_dict['precision'] = precision_score(true_labels, pred_labels, average=average, pos_label=pos_label)
        metrics_dict['recall'] = recall_score(true_labels, pred_labels, average=average, pos_label=pos_label)
        metrics_dict['sensitivity'] = recall_score(true_labels, pred_labels, average=average, pos_label=pos_label)
        metrics_dict['specificity'] = recall_score(true_labels, pred_labels, average=average, pos_label=1-pos_label)

        return metrics_dict
