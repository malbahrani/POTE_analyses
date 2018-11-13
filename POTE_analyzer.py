import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import gridspec

## Read in LINE-element data
df_line = pd.read_csv('./data/outputs/hg38_LINEs_within_POTEs_10K_flankings__RefTable__.txt', sep='\t')

## Read in CpGs within POTEs data 
df_cpg = pd.read_csv('./data/outputs/450K_probes_within_POTEs.txt', sep='\t', header=None)
df_cpg.columns = ['chrom_gene', 'start_gene', 'end_gene', 'gene_id', 'number_gene', 'strand_gene', 'chrom_cpg', 'start_cpg', 'end_cpg', 'cpg_id', 'number_cpg', 'strand_cpg']

## Read in reference tables.
# Methylation.
case_ref = pd.read_csv('./data/patch_GSE65820/Patch_GEO_Patient_Information.txt', sep='\t')
case_ref['Source name'].replace('Primarytumour', 'PrimaryTumour', inplace=True)
case_ref['Tissue'].replace('Primarytumour', 'PrimaryTumour', inplace=True)
meth_ref1 = pd.read_csv('./data/patch_GSE65820/GSE65821_GEO_to_DCC_IDs.txt.gz', compression='gzip', sep='\t')
meth_ref1['extracted_sample_id'] = meth_ref1.submitted_sample_id.str.extract('(\w{4}\-\w{3})')
meth_ref2 = pd.read_csv('./data/patch_GSE65820/GEO_samples_table.txt', sep='\t')
# join meth_ref1 with meth_ref2 on GEO_sample_id
meth_ref  =  meth_ref1.merge(meth_ref2, left_on='GEO_sample_id', right_on='GEO_sample_id', how='inner')
# Expression.
specimen_ref = pd.read_csv('./data/Patch_HGSC_expression/Bowtell_specimen_IDs.tsv', sep='\t')
expre_ref = pd.read_csv('./data/Patch_HGSC_expression/Bowtell_POTEs_exp_seq.tsv', sep='\t', header=None)
expre_ref.columns = list(pd.read_csv('./data/Patch_HGSC_expression/Bowtell_exp_seq__header___.tsv', sep='\t').columns)
expre_ref['extracted_sample_id'] = expre_ref.submitted_sample_id.str.extract('(\w{4}\-\w{3})')

## Read in methyl data.
meth_POTEs_ov = pd.read_csv('./data/outputs/450K_probes_within_POTEs_30K_flankings__Patch_OV_Data__.csv')
meth_POTEs_ft = pd.read_csv('./data/outputs/450K_probes_within_POTEs_30K_flankings__Patch_FT_Data__.csv')

## Read in expression data.
expre = pd.read_csv('./data/Patch_HGSC_expression/Bowtell_POTEs_exp_seq.tsv', sep='\t')
expre.columns = list(pd.read_csv('./data/Patch_HGSC_expression/Bowtell_exp_seq__header___.tsv', sep='\t').columns)
expre['extracted_sample_id'] = expre.submitted_sample_id.str.extract('(AOCS-\d{3})', expand=False)

def L1_pinpointer(L1_table):
    '''
    Create a table that count the number of LINE-1 elements within each POTE.
    '''
    return L1_table.groupby(['POTE_id']).sum()[['LINE_bases_overlapped_with_POTE_flanks']]
def cpg_upstream_pinpointer(cpg_table, bases):
    '''
    Create a table that lists the cpgs upstream POTE genes' TSS .
    
    INPUTS:
    cpg_table : dataframe object.
    bases     : int object specifies the number of bases upstream of POTE to find the CpGs within.
    
    OUTPUTS:
    dataframe object
    '''
    listy = []
    for gene in list(cpg_table.groupby('gene_id')):
        #print('processing', gene[0])
        if gene[1].strand_gene.unique() == '+':
            listy.append(gene[1][(gene[1].start_cpg > int(gene[1].start_gene.unique()-bases)) & (gene[1].start_cpg < int(gene[1].start_gene.unique()))][['gene_id','cpg_id']])       
        else:
            listy.append(gene[1][(gene[1].start_cpg < int(gene[1].end_gene.unique()+bases)) & (gene[1].start_cpg > int(gene[1].end_gene.unique()))][['gene_id','cpg_id']])       

    return pd.concat(listy)   
def cpg_body_pinpointer(cpg_table):
    '''
    Create a table that lists the cpgs within POTE genes' body.
    
    INPUTS:
    cpg_table : dataframe object.
    
    OUTPUTS:
    dataframe
    '''
    listy = []
    for gene in list(cpg_table.groupby('gene_id')):
        #print('processing', gene[0])
        listy.append(gene[1][(gene[1].start_cpg > int(gene[1].start_gene.unique())) & (gene[1].start_cpg < int(gene[1].end_gene.unique()))][['gene_id','cpg_id']])       
    return pd.concat(listy)
def compare_fte_to_prim_methyl(df_ft, df_ov, sort_cpg_by ,name):
    '''
    Generate the average CpG sites methylation of the input POTE from FTE and primary samples.

    INPUTS:
    
    df_ft: DataFrame object. CpG probes as columns, and fallobian tube samples as rows
    df_ov: DataFrame object. CpG probes as columns, and High-grade serous ovarian cancer samples as rows.
    
    sort_cpg_by: string object; specifies the column label by which the DataFrame object will be sorted.
    name       : string object; specifies the gene symbol (eg. POTEC).
    
    OUTPUTS:
    df_ft_and_ov        : DataFrame object. Methylation matrix with CpG methylation (averaged across samples) for FTE and HGSC (primary only).
    df_ft_and_ov_change : DataFrame object. One row matrix showing the difference in methylation between FTE and HGSC (primary only) for each CpG. 
    '''
    # Process FTE methylation object
    # choose methylation records for the input gene symbol
    df_ft_cpg_within_probe = df_ft[df_ft.name.str.contains(name)].sort_values(sort_cpg_by).reset_index(drop=True)
    df_ft_cpg_within_probe.reset_index(inplace=True)
    order = list(df_ft_cpg_within_probe['ID_REF'])
    df_ft_cpg_within_probe = df_ft[df_ft.name.str.contains(name)].iloc[:,-7:].set_index('ID_REF')
    df_ft_cpg_within_probe = df_ft_cpg_within_probe.reindex(index=order).T
    # Replace missing CpG methylation intensities with the mean values of the CpGs across the samples.
    df_ft_cpg_within_probe.fillna(df_ft_cpg_within_probe.mean(), inplace=True)
    df_ft_cpg_within_probe = df_ft_cpg_within_probe.astype(np.float)
    # Average CpG methylation across FTE samples
    df_ft_cpg_within_probe_ave = df_ft_cpg_within_probe.mean().to_frame()
    df_ft_cpg_within_probe_ave = df_ft_cpg_within_probe_ave.reset_index()
    df_ft_cpg_within_probe_ave.columns = ['CpG', 'FTE']
    df_ft_cpg_within_probe_ave.set_index('CpG', inplace=True)
 
    # Process HGSC methylation object
    df_ov_cpg_within_probe = df_ov[df_ov.name.str.contains(name)].sort_values(sort_cpg_by).reset_index(drop=True)
    df_ov_cpg_within_probe.reset_index(inplace=True)
    order = list(df_ov_cpg_within_probe['ID_REF'])
    df_ov_cpg_within_probe = df_ov[df_ov.name.str.contains(name)].iloc[:,-114:].set_index('ID_REF')
    df_ov_cpg_within_probe = df_ov_cpg_within_probe.reindex(index=order).T
    df_ov_cpg_within_probe = df_ov_cpg_within_probe.loc[case_ref[case_ref['Source name'] == 'PrimaryTumour'].merge(meth_ref2, left_on='Accession_Methylation', right_on='GEO_GSM_ID', how='inner').Accession_Methylation,:]
    # Replace missing CpG methylation intensities with the mean values of the CpGs across the samples.
    df_ov_cpg_within_probe.fillna(df_ov_cpg_within_probe.mean(), inplace=True)
    df_ov_cpg_within_probe = df_ov_cpg_within_probe.astype(np.float)
      
    # Average CpG methylation across HGSC samples.
    df_ov_cpg_within_probe_ave = df_ov_cpg_within_probe.mean().to_frame()
    df_ov_cpg_within_probe_ave = df_ov_cpg_within_probe_ave.reset_index()
    df_ov_cpg_within_probe_ave.columns = ['CpG', 'HGSC']
    df_ov_cpg_within_probe_ave.set_index('CpG', inplace=True)
    
    # Concat. FTE and HGSC methylation data into one object.
    df_ft_and_ov = pd.concat([df_ft_cpg_within_probe_ave, df_ov_cpg_within_probe_ave], axis=1).T
    df_ft_and_ov_labels = df_ft_and_ov.columns
    
    # Calculate the difference in methylation between FTE and HGSC.
    df_ft_and_ov_change = (df_ft_and_ov.iloc[0, :] - df_ft_and_ov.iloc[1, :])*(-100)

    return df_ft_and_ov, df_ft_and_ov_change, df_ov_cpg_within_probe_ave
def compare_prim_match_to_recur_match_methyl(df_case_ref, df_meth, pote_name, sort_arg):
    '''
    Generate the average CpG sites methylation of the input POTE from patient-matched primary and recurrent samples.

    INPUTS:
    df_case_ref     : DataFrame object. Cases meta-data.
    df_meth         : DataFrame object. semi-processed data.
    pote_name       : String object. Gene symbol (e.g. POTEC).
    sort_arg        : Boolen.  True= ascending; otherwise descending.

    OUTPUTS:
    df_prim_and_recur_metch         : DataFrame object. Methylation matrix with CpG methylation (averaged across samples) for patient matched primary and recurrent HGSC.
    df_prim_and_recur_change        : Series object. Shows the difference in methylation between patient matched primary and recurrent HGSC for each CpG site.

    '''
    ## Process Methylation data set
    # Extract the ensembl ID from `pote_name` input arg (gene symbol).
    pote_ensembl_id = list(df_meth[df_meth.name.str.contains(pote_name)].name.str.split('_').str[1])[0] 
    # CpG probs as columns; rows are methylation samples ids
    df_meth_mx = df_meth[df_meth.name.str.contains(pote_name)].sort_values('cpg_start', ascending=sort_arg).iloc[:,-114:].set_index('ID_REF').T
    # Drop row instance where all values are NaNs.
    df_meth_mx.dropna(axis=0, how='all', inplace=True)
    df_meth_mx.reset_index(inplace=True)
    # Create `ID_REF_col` as the primary key for each methylation sample (GSM ids)
    df_meth_mx.rename({'index':'ID_REF_col'}, axis=1, inplace=True)
    # Join methylation 450K data with reference table `case_ref`    
    df_meth_mx_withREF = df_meth_mx.merge(df_case_ref[['Accession_Methylation', 'Title', 'Source name']], left_on='ID_REF_col', right_on='Accession_Methylation', how='inner')
    # Extract case ids from `Title` column
    df_meth_mx_withREF['extracted_sample_id'] = df_meth_mx_withREF.Title.str.extract('(\w{4}\_\w{3})')
    df_meth_mx_withREF.extracted_sample_id = df_meth_mx_withREF.extracted_sample_id.str.replace('\_','-')
    df_meth_mx_withREF.set_index(['extracted_sample_id', 'ID_REF_col', 'Source name'], inplace=True)
    df_meth_mx_withREF.sort_index(inplace=True)
    df_meth_mx_withREF = df_meth_mx_withREF[df_meth_mx_withREF.columns[df_meth_mx_withREF.columns.str.startswith('cg')]]
    df_meth_mx_withREF = df_meth_mx_withREF.reset_index()
    # Create sets methyl (primary) and methyl (recurrent)
    # 79 samples
    df_meth_mx_withREF_prim  = df_meth_mx_withREF[df_meth_mx_withREF['Source name'] == 'PrimaryTumour']
    # fill NaNs
    df_meth_mx_withREF_prim.fillna(df_meth_mx_withREF_prim.mean(), inplace=True)    
    # 34 samples
    df_meth_mx_withREF_recur = df_meth_mx_withREF[df_meth_mx_withREF['Source name'] != 'PrimaryTumour']
    # fill NaNs
    df_meth_mx_withREF_recur.fillna(df_meth_mx_withREF_recur.mean(), inplace=True)
    # Average sample replicates
    df_meth_mx_withREF_prim = df_meth_mx_withREF_prim.groupby('extracted_sample_id').mean().reset_index()
    df_meth_mx_withREF_recur= df_meth_mx_withREF_recur.groupby('extracted_sample_id').mean().reset_index()
    # 13 primary recurrent matched sample (belong to the same patient)
    meth_prim_recur_match_AOCS_ids = np.intersect1d(df_meth_mx_withREF_prim.extracted_sample_id, df_meth_mx_withREF_recur.extracted_sample_id, assume_unique=False)
    ## Select cases with matched primary and recurrent tumours
    df_meth_mx_withREF_prim = df_meth_mx_withREF_prim.set_index('extracted_sample_id').loc[meth_prim_recur_match_AOCS_ids,:]
    df_meth_mx_withREF_recur = df_meth_mx_withREF_recur.set_index('extracted_sample_id').loc[meth_prim_recur_match_AOCS_ids,:]
     # Average CpG methylation across HGSC samples.
    df_prim_cpg_within_probe_ave = df_meth_mx_withREF_prim.mean().to_frame()
    df_prim_cpg_within_probe_ave = df_prim_cpg_within_probe_ave.reset_index()
    df_prim_cpg_within_probe_ave.columns = ['CpG', 'HGSC(primary)']
    df_prim_cpg_within_probe_ave.set_index('CpG', inplace=True)
    
    df_recur_cpg_within_probe_ave = df_meth_mx_withREF_recur.mean().to_frame()
    df_recur_cpg_within_probe_ave = df_recur_cpg_within_probe_ave.reset_index()
    df_recur_cpg_within_probe_ave.columns = ['CpG', 'HGSC(recurrent)']
    df_recur_cpg_within_probe_ave.set_index('CpG', inplace=True)
    
    # Concat. primary and recurrent matched HGSC methylation data into one object.
    df_prim_and_recur = pd.concat([df_prim_cpg_within_probe_ave, df_recur_cpg_within_probe_ave], axis=1).T
   
    # Calculate the difference in methylation between primary and recurrent matched HGSC.
    df_prim_and_recur_change = (df_prim_and_recur.iloc[0, :] - df_prim_and_recur.iloc[1, :])*(-100)

    return df_prim_and_recur, df_prim_and_recur_change
def compare_prim_match_to_recur_match_methyl_and_expre(df_case_ref, df_meth, df_specimen_ref ,df_expre, pote_name, cpgs_index_range, sort_arg):
    '''
    Find the patients primary and recurrent match samples which have been profiled for methylation and expression.
    '''
    ## Process Methylation data set
    # Extract the ensembl ID from `pote_name` input arg (gene symbol).
    pote_ensembl_id = list(df_meth[df_meth.name.str.contains(pote_name)].name.str.split('_').str[1])[0] 
    # CpG probs as columns; rows are methylation samples ids
    df_meth_mx = df_meth[df_meth.name.str.contains(pote_name)].sort_values('cpg_start', ascending=sort_arg).iloc[:,-114:].set_index('ID_REF').T
    # Drop row instance where all values are NaNs.
    df_meth_mx.dropna(axis=0, how='all', inplace=True)
    df_meth_mx.reset_index(inplace=True)
    # Create `ID_REF_col` as the primary key for each methylation sample (GSM ids)
    df_meth_mx.rename({'index':'ID_REF_col'}, axis=1, inplace=True)
    # Join methylation 450K data with reference table `case_ref`    
    df_meth_mx_withREF = df_meth_mx.merge(df_case_ref[['Accession_Methylation', 'Title', 'Source name']], left_on='ID_REF_col', right_on='Accession_Methylation', how='inner')
    # Extract case ids from `Title` column
    df_meth_mx_withREF['extracted_sample_id'] = df_meth_mx_withREF.Title.str.extract('(\w{4}\_\w{3})')
    df_meth_mx_withREF.extracted_sample_id = df_meth_mx_withREF.extracted_sample_id.str.replace('\_','-')
    df_meth_mx_withREF.set_index(['extracted_sample_id', 'ID_REF_col', 'Source name'], inplace=True)
    df_meth_mx_withREF.sort_index(inplace=True)
    df_meth_mx_withREF = df_meth_mx_withREF[df_meth_mx_withREF.columns[df_meth_mx_withREF.columns.str.startswith('cg')]]
    df_meth_mx_withREF = df_meth_mx_withREF.reset_index()
    # Create sets methyl (primary) and methyl (recurrent)
    # 79 samples
    df_meth_mx_withREF_prim  = df_meth_mx_withREF[df_meth_mx_withREF['Source name'] == 'PrimaryTumour']
    # fill NaNs
    df_meth_mx_withREF_prim.fillna(df_meth_mx_withREF_prim.mean(), inplace=True)    
    # 34 samples
    df_meth_mx_withREF_recur = df_meth_mx_withREF[df_meth_mx_withREF['Source name'] != 'PrimaryTumour']
    # fill NaNs
    df_meth_mx_withREF_recur.fillna(df_meth_mx_withREF_recur.mean(), inplace=True)
    # Average sample replicates
    df_meth_mx_withREF_prim = df_meth_mx_withREF_prim.groupby('extracted_sample_id').mean().reset_index()
    df_meth_mx_withREF_recur= df_meth_mx_withREF_recur.groupby('extracted_sample_id').mean().reset_index()
    # 13 primary recurrent matched sample (belong to the same patient)
    meth_prim_recur_match_AOCS_ids = np.intersect1d(df_meth_mx_withREF_prim.extracted_sample_id, df_meth_mx_withREF_recur.extracted_sample_id, assume_unique=False)
    
    ## Process Expression data set:
    # select the needed column.
    df_specimen_ref_sub = df_specimen_ref[['icgc_donor_id','icgc_specimen_id','submitted_donor_id', 'specimen_type']]
    # select the needed column from the expression table.
    df_expre_extract    = df_expre[['icgc_donor_id','icgc_specimen_id','extracted_sample_id', 'gene_id', 'normalized_read_count']]
    df_expre_mx_withREF = df_expre_extract.merge(df_specimen_ref_sub, left_on=['icgc_donor_id', 'icgc_specimen_id'], right_on=['icgc_donor_id','icgc_specimen_id'], how='inner')
    # select primary tumours
    df_expre_mx_withREF_prim = df_expre_mx_withREF[df_expre_mx_withREF['specimen_type'] == 'Primary tumour - solid tissue']
    # select recurrent tumours
    df_expre_mx_withREF_recur= df_expre_mx_withREF[df_expre_mx_withREF['specimen_type'].isin(['Recurrent tumour - solid tissue', 'Recurrent tumour - other', 'Metastatic tumour - metastasis to distant location'])]
    # select the POTE gene
    df_expre_mx_withREF_prim_pote  = df_expre_mx_withREF_prim[df_expre_mx_withREF_prim.gene_id == pote_ensembl_id]
    df_expre_mx_withREF_recur_pote = df_expre_mx_withREF_recur[df_expre_mx_withREF_recur.gene_id == pote_ensembl_id]
    # average replicates readouts of POTE  for a given case
    # 81 total cases with primary tumours profiled for the gene (replicates have been averaged)
    df_expre_mx_withREF_prim_pote  = df_expre_mx_withREF_prim_pote.groupby(['icgc_donor_id', 'submitted_donor_id']).mean().reset_index()
    df_expre_mx_withREF_prim_pote.set_index('submitted_donor_id', inplace=True)
    # 24 total cases with recurrent tumours profiled for the gene (replicates have been averaged)
    df_expre_mx_withREF_recur_pote = df_expre_mx_withREF_recur_pote.groupby(['icgc_donor_id', 'submitted_donor_id']).mean().reset_index()
    df_expre_mx_withREF_recur_pote.set_index('submitted_donor_id', inplace=True)
    # 11 primary recurrent matched sample (belong to the same patient)
    expre_prim_recur_match_AOCS_ids = np.intersect1d(df_expre_mx_withREF_prim_pote.index, df_expre_mx_withREF_recur_pote.index, assume_unique=False)
    ## Determine the most inner matched set 
    # 10 primary recurrent matched sample profiled for methylation and expression 
    meth_expre_prim_recur_match_AOCS_ids = np.intersect1d(meth_prim_recur_match_AOCS_ids, expre_prim_recur_match_AOCS_ids, assume_unique=False)
    ## Select cases with matched primary and recurrent tumours
    # Methylation:
    # Select CpGs near TSS (taken as input in the function)     
    df_meth_mx_withREF_prim = df_meth_mx_withREF_prim.set_index('extracted_sample_id').loc[meth_expre_prim_recur_match_AOCS_ids,:].iloc[:,cpgs_index_range]
    df_meth_mx_withREF_prim['average_methylation'] = df_meth_mx_withREF_prim.mean(axis=1)
    df_meth_mx_withREF_prim_averaged_cpgs = df_meth_mx_withREF_prim[['average_methylation']]
    df_meth_mx_withREF_prim_averaged_cpgs['sample'] = 'Primary' 
    df_meth_mx_withREF_recur = df_meth_mx_withREF_recur.set_index('extracted_sample_id').loc[meth_expre_prim_recur_match_AOCS_ids,:].iloc[:,cpgs_index_range]
    df_meth_mx_withREF_recur['average_methylation'] = df_meth_mx_withREF_recur.mean(axis=1)
    df_meth_mx_withREF_recur_averaged_cpgs = df_meth_mx_withREF_recur[['average_methylation']]
    df_meth_mx_withREF_recur_averaged_cpgs['sample'] = 'Recurrent'
    # concat
    meth_concat = pd.concat([df_meth_mx_withREF_prim_averaged_cpgs, df_meth_mx_withREF_recur_averaged_cpgs], axis=0)
    meth_concat['experiment'] = 'methylation'
    meth_concat.rename({'average_methylation':'value'}, axis=1, inplace=True)
    meth_concat.reset_index(inplace=True)
    meth_concat.rename({'extracted_sample_id':'Case'}, axis=1, inplace=True)
    # Expression:
    # Select the cases
    df_expre_mx_withREF_prim_pote  = df_expre_mx_withREF_prim_pote.loc[meth_expre_prim_recur_match_AOCS_ids,:]
    df_expre_mx_withREF_prim_pote['sample'] = 'Primary'
    df_expre_mx_withREF_recur_pote = df_expre_mx_withREF_recur_pote.loc[meth_expre_prim_recur_match_AOCS_ids,:]
    df_expre_mx_withREF_recur_pote['sample'] = 'Recurrent'
    # concat
    expre_concat = pd.concat([df_expre_mx_withREF_prim_pote, df_expre_mx_withREF_recur_pote], axis=0)
    expre_concat['experiment'] = 'expression'
    expre_concat.drop('icgc_donor_id', axis=1, inplace=True)
    expre_concat.rename({'normalized_read_count': 'value'}, axis=1, inplace=True)
    expre_concat.reset_index(inplace=True)
    expre_concat.rename({'submitted_donor_id':'Case'}, axis=1, inplace=True)

    
    #meth_expre = pd.concat([meth_concat, expre_concat], axis=0)
    #meth_expre['label'] = meth_expre[['experiment', 'sample']].apply(lambda x: '_'.join(x), axis=1)
    #meth_expre.drop(['experiment', 'sample'], axis=1, inplace=True)
    
    #return meth_expre
    return meth_concat, expre_concat
def methyl_expre_integrator(df_meth, df_meth_ref, df_expre, pote_name, cpgs_index_range, sort_arg):
    '''
    Generate methylation and expression matrices for the input POTE.  The data come from primary and recurrent tumours.

    INPUTS:
    df_meth  : DataFrame object. semi-processed data.
    df_expre : DataFrame object. semi-processed data.
    pote_name: String object. Gene symbol (e.g. POTEC)
    sort_arg : Boolen.  True= ascending; otherwise descending.
    cpgs_index_range : []
    
    OUTPUTS:
    df_meth_mx                : Columns CpG probes.  Rows samples of HGSC.
    df_expre_transform_scaled : One column representing the expression of the specified POTE acros
    '''
    # Process Methyl. table.
    # Drop methylation sample that does not have data submitted in GEO
    df_meth_ref = df_meth_ref.drop(44, axis=0)
    # Rename methylation samples' label to matching names with RNA-seq samples' labels 
    df_meth.rename(columns = dict(zip(df_meth.columns[-113:], list(df_meth_ref.extracted_sample_id))), inplace=True)
    # Fill missing value in the rows with mean of row (HGSC data sets)
    df_meth.iloc[:,-113:] = df_meth.iloc[:,-113:].T.fillna(df_meth.iloc[:,-113:].mean(axis=1)).T

    pote_ensembl_id = list(df_meth[df_meth.name.str.contains(pote_name)].name.str.split('_').str[1])[0]
    df_meth_mx = df_meth[df_meth.name.str.contains(pote_name)].sort_values('cpg_start', ascending=sort_arg).iloc[:,-114:].set_index('ID_REF').T
    # Average replicated samples in 450K dataFrame
    df_meth_mx= df_meth_mx.groupby(df_meth_mx.index).mean()
    df_meth_mx.reset_index(inplace=True)
    df_meth_mx.rename(columns={'index':'ID_REF'}, inplace=True)

    # Compute the average POTE expression for replicated samples
    df_expre_average_replicates = df_expre.groupby(['extracted_sample_id', 'gene_id'])['normalized_read_count'].mean().to_frame().reset_index()
    
    # Integrate methyl. and expre. samples.
    df_meth_expre_mx = df_meth_mx.merge(df_expre_average_replicates[df_expre_average_replicates.gene_id == pote_ensembl_id], how='inner', left_on='ID_REF', right_on='extracted_sample_id')
    df_meth_expre_mx['ID'] = df_meth_expre_mx['extracted_sample_id'] + '_' + df_meth_expre_mx['gene_id']
    df_meth_expre_mx.drop(['ID_REF', 'extracted_sample_id', 'gene_id'], axis=1, inplace=True)
    df_meth_expre_mx.set_index('ID', inplace=True)

    # Process Expre. column. 
    # Slice expression column
    df_expre_mx = df_meth_expre_mx.iloc[:, -1:]
    # Filter samples with zero expression values (Because zeros cause issues when transforming the data to log2)
    df_expre_mx = df_expre_mx[df_expre_mx.normalized_read_count > 0]
    # Apply log2 transformation to make the expression close to normal distribution.
    df_expre_mx_transform = np.log2(df_expre_mx)
    expre_mx_labels = df_expre_mx_transform.columns
    expre_mx_index = df_expre_mx_transform.index
    # Scale the data to -2:2 interval.
    df_expre_mx_transform_scaled = pd.DataFrame(preprocessing.minmax_scale(df_expre_mx_transform, feature_range=(-2,2)))
    df_expre_mx_transform_scaled.columns = expre_mx_labels
    df_expre_mx_transform_scaled.index = expre_mx_index
    
    # Filter the methylation matching samples that have zero as expression values.
    df_meth_mx = df_meth_expre_mx.iloc[:,:-1]*100
    df_meth_mx = df_meth_mx.loc[df_expre_mx_transform.index,:]
    
    return (df_meth_mx.iloc[:, cpgs_index_range], df_expre_mx_transform_scaled)

    ## L1 and CpGs summary table
line     = L1_pinpointer(df_line)
cpg_30k  = cpg_upstream_pinpointer(df_cpg, 30000).rename({'cpg_id':'cpg_30K_up_TSS'}, axis=1).set_index(['gene_id','cpg_30K_up_TSS'])
cpg_10k  = cpg_upstream_pinpointer(df_cpg, 10000).rename({'cpg_id':'cpg_10K_up_TSS'}, axis=1).set_index(['gene_id','cpg_10K_up_TSS'])
cpg_body = cpg_body_pinpointer(df_cpg).rename({'cpg_id':'cpg_within_pote'}, axis=1).set_index(['gene_id', 'cpg_within_pote'])

writer1 = pd.ExcelWriter('./data/outputs/cpg_probes_ids_within_potes.xls')
cpg_30k.reset_index().to_excel(writer1, 'CpG_30K_upstream', index=False)
cpg_10k.reset_index().to_excel(writer1, 'Cpg_10K_upstream', index=False)
cpg_body.reset_index().to_excel(writer1, 'Cpg_within_body', index=False)
writer1.save()

## [Active]
writer = pd.ExcelWriter('./data/outputs/L1_cpg_summary.xls')
pd.concat([line, cpg_30k.reset_index().groupby('gene_id').count(), 
           cpg_10k.reset_index().groupby('gene_id').count()
           , cpg_body.reset_index().groupby('gene_id').count()], axis=1).to_excel(writer, 'counts')
writer.save()

%matplotlib agg 
# magic linke to not show any inline output
##Analysis.1
print('[Begin] Analysis #1 [untitled]')
# POTEA
print('[Run on] POTEA')
potea_fte_prim              = compare_fte_to_prim_methyl(meth_POTEs_ft, meth_POTEs_ov, 'cpg_start', 'POTEA')[0] # profile object.
potea_fte_prim_diff         = compare_fte_to_prim_methyl(meth_POTEs_ft, meth_POTEs_ov, 'cpg_start', 'POTEA')[1] # difference.
potea_prim_recur_match      = compare_prim_match_to_recur_match_methyl(case_ref, meth_POTEs_ov, 'POTEA', True)[0] # profile object.
potea_prim_recur_match_diff = compare_prim_match_to_recur_match_methyl(case_ref, meth_POTEs_ov, 'POTEA', True)[1] # difference. 
#POTEC
print('[Run on] POTEC')
potec_fte_prim              = compare_fte_to_prim_methyl(meth_POTEs_ft, meth_POTEs_ov, 'cpg_start', 'POTEC')[0] # profile object.
potec_fte_prim_diff         = compare_fte_to_prim_methyl(meth_POTEs_ft, meth_POTEs_ov, 'cpg_start', 'POTEC')[1] # difference.
potec_prim_recur_match      = compare_prim_match_to_recur_match_methyl(case_ref, meth_POTEs_ov, 'POTEC', True)[0] # profile object.
potec_prim_recur_match_diff = compare_prim_match_to_recur_match_methyl(case_ref, meth_POTEs_ov, 'POTEC', True)[1] # difference. 
#POTEE
print('[Run on] POTEE')
potee_fte_prim              = compare_fte_to_prim_methyl(meth_POTEs_ft, meth_POTEs_ov, 'cpg_start', 'POTEE')[0] # profile object.
potee_fte_prim_diff         = compare_fte_to_prim_methyl(meth_POTEs_ft, meth_POTEs_ov, 'cpg_start', 'POTEE')[1] # difference.
potee_prim_recur_match      = compare_prim_match_to_recur_match_methyl(case_ref, meth_POTEs_ov, 'POTEE', True)[0] # profile object.
potee_prim_recur_match_diff = compare_prim_match_to_recur_match_methyl(case_ref, meth_POTEs_ov, 'POTEE', True)[1] # difference. 
#POTEF
print('[Run on] POTEF')
potef_fte_prim              = compare_fte_to_prim_methyl(meth_POTEs_ft, meth_POTEs_ov, 'cpg_start', 'POTEF')[0] # profile object.
potef_fte_prim_diff         = compare_fte_to_prim_methyl(meth_POTEs_ft, meth_POTEs_ov, 'cpg_start', 'POTEF')[1] # difference.
potef_prim_recur_match      = compare_prim_match_to_recur_match_methyl(case_ref, meth_POTEs_ov, 'POTEF', True)[0] # profile object.
potef_prim_recur_match_diff = compare_prim_match_to_recur_match_methyl(case_ref, meth_POTEs_ov, 'POTEF', True)[1] # difference. 
print('\n\n')

print('[Vizualize] Analysis #1 [untitled]')
print('[Begin] A')
# ************ POTEA ************
fig = plt.figure(figsize=(381.5,35))
sns.set(font_scale=12)
gs = gridspec.GridSpec(2,1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
fig.subplots_adjust(hspace=0)

# Plot heatmap with no colors (Table figure)
with sns.axes_style('white'):
    sns.heatmap(potea_fte_prim, cbar=False, square=False, xticklabels=False, yticklabels=['FTE', 'HGSC (Primary)'], annot=True, annot_kws={"weight":'bold'}, fmt='.3f', linecolor='black', linewidths=20, cmap=ListedColormap(['white']), ax=ax1)

ax = sns.heatmap(potea_fte_prim_diff.to_frame().T, cbar=False, square=False, xticklabels=True, yticklabels=['% difference'], annot=True, annot_kws={"weight":'bold'}, fmt= '.0f' ,linecolor='black', linewidths=20, cmap='OrRd_r', ax=ax2)
for t in ax.texts: t.set_text(t.get_text() + " %")
plt.xticks(rotation=45, weight='bold', fontsize='medium')
plt.yticks(rotation='horizontal')
ax.set_xlabel('')
# Save figure as .svg file.
plt.savefig('./potea_methyl_percent_primary_vs_fte.svg', format='svg', bbox_inches='tight', pad_inches=0)
print('[Save] ./potea_methyl_percent_primary_vs_fte.svg')

fig = plt.figure(figsize=(381.5,35))
sns.set(font_scale=12)
gs = gridspec.GridSpec(2,1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
fig.subplots_adjust(hspace=0)

# Plot heatmap with no colors (Table figure)
with sns.axes_style('white'):
    sns.heatmap(potea_prim_recur_match, cbar=False, square=False, xticklabels=False, yticklabels=['HGSC (Primary)', 'HGSC (Recurrent)'], annot=True, annot_kws={"weight":'bold'}, fmt='.3f', linecolor='black', linewidths=20, cmap=ListedColormap(['white']), ax=ax1)

ax = sns.heatmap(potea_prim_recur_match_diff.to_frame().T, cbar=False, square=False, xticklabels=True, yticklabels=['% difference'], annot=True, annot_kws={"weight":'bold'}, fmt= '.0f' ,linecolor='black', linewidths=20, cmap='OrRd_r', ax=ax2)
for t in ax.texts: t.set_text(t.get_text() + " %")
plt.xticks(rotation=45, weight='bold', fontsize='medium')
plt.yticks(rotation='horizontal')
ax.set_xlabel('')
# Save figure as .svg file.
plt.savefig('./potea_methyl_percent_primary_vs_recurrent.svg', format='svg', bbox_inches='tight', pad_inches=0)
print('[Save] ./potea_methyl_percent_primary_vs_recurrent.svg \n')

print('[Begin] C')
# ************ POTEC ************
fig = plt.figure(figsize=(160,35))
sns.set(font_scale=12)
gs = gridspec.GridSpec(2,1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
fig.subplots_adjust(hspace=0)

# Plot heatmap with no colors (Table figure)
with sns.axes_style('white'):
    sns.heatmap(potec_fte_prim, cbar=False, square=False, xticklabels=False, yticklabels=['FTE', 'HGSC (Primary)'], annot=True, annot_kws={"weight":'bold'}, fmt='.3f', linecolor='black', linewidths=20, cmap=ListedColormap(['white']), ax=ax1)

ax = sns.heatmap(potec_fte_prim_diff.to_frame().T, cbar=False, square=False, xticklabels=True, yticklabels=['% difference'], annot=True, annot_kws={"weight":'bold'}, fmt= '.0f' ,linecolor='black', linewidths=20, cmap='OrRd_r', ax=ax2)
for t in ax.texts: t.set_text(t.get_text() + " %")
plt.xticks(rotation=45, weight='bold', fontsize='medium')
plt.yticks(rotation='horizontal')
ax.set_xlabel('')
# Save figure as .svg file.
plt.savefig('./potec_methyl_percent_primary_vs_fte.svg', format='svg', bbox_inches='tight', pad_inches=0)
print('[Save] ./potec_methyl_percent_primary_vs_fte.svg')

fig = plt.figure(figsize=(160,35))
sns.set(font_scale=12)
gs = gridspec.GridSpec(2,1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
fig.subplots_adjust(hspace=0)

# Plot heatmap with no colors (Table figure)
with sns.axes_style('white'):
    sns.heatmap(potec_prim_recur_match, cbar=False, square=False, xticklabels=False, yticklabels=['HGSC (Primary)', 'HGSC (Recurrent)'], annot=True, annot_kws={"weight":'bold'}, fmt='.3f', linecolor='black', linewidths=20, cmap=ListedColormap(['white']), ax=ax1)

ax = sns.heatmap(potec_prim_recur_match_diff.to_frame().T, cbar=False, square=False, xticklabels=True, yticklabels=['% difference'], annot=True, annot_kws={"weight":'bold'}, fmt= '.0f' ,linecolor='black', linewidths=20, cmap='OrRd_r', ax=ax2)
for t in ax.texts: t.set_text(t.get_text() + " %")
plt.xticks(rotation=45, weight='bold', fontsize='medium')
plt.yticks(rotation='horizontal')
ax.set_xlabel('')
# Save figure as .svg file.
plt.savefig('./potec_methyl_percent_primary_vs_recurrent.svg', format='svg', bbox_inches='tight', pad_inches=0)
print('[Save] ./potec_methyl_percent_primary_vs_recurrent.svg \n')

print('[Begin] E')
# ************ POTEE ************
fig = plt.figure(figsize=(135.38,35))
sns.set(font_scale=12)
gs = gridspec.GridSpec(2,1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
fig.subplots_adjust(hspace=0)

# Plot heatmap with no colors (Table figure)
with sns.axes_style('white'):
    sns.heatmap(potee_fte_prim, cbar=False, square=False, xticklabels=False, yticklabels=['FTE', 'HGSC (Primary)'], annot=True, annot_kws={"weight":'bold'}, fmt='.3f', linecolor='black', linewidths=20, cmap=ListedColormap(['white']), ax=ax1)

ax = sns.heatmap(potee_fte_prim_diff.to_frame().T, cbar=False, square=False, xticklabels=True, yticklabels=['% difference'], annot=True, annot_kws={"weight":'bold'}, fmt= '.0f' ,linecolor='black', linewidths=20, cmap='OrRd_r', ax=ax2)
for t in ax.texts: t.set_text(t.get_text() + " %")
plt.xticks(rotation=45, weight='bold', fontsize='medium')
plt.yticks(rotation='horizontal')
ax.set_xlabel('')
# Save figure as .svg file.
plt.savefig('./potee_methyl_percent_primary_vs_fte.svg', format='svg', bbox_inches='tight', pad_inches=0)
print('[Save] ./potee_methyl_percent_primary_vs_fte.svg')

fig = plt.figure(figsize=(135.38,35))
sns.set(font_scale=12)
gs = gridspec.GridSpec(2,1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
fig.subplots_adjust(hspace=0)

# Plot heatmap with no colors (Table figure)
with sns.axes_style('white'):
    sns.heatmap(potee_prim_recur_match, cbar=False, square=False, xticklabels=False, yticklabels=['HGSC (Primary)', 'HGSC (Recurrent)'], annot=True, annot_kws={"weight":'bold'}, fmt='.3f', linecolor='black', linewidths=20, cmap=ListedColormap(['white']), ax=ax1)

ax = sns.heatmap(potee_prim_recur_match_diff.to_frame().T, cbar=False, square=False, xticklabels=True, yticklabels=['% difference'], annot=True, annot_kws={"weight":'bold'}, fmt= '.0f' ,linecolor='black', linewidths=20, cmap='OrRd_r', ax=ax2)
for t in ax.texts: t.set_text(t.get_text() + " %")
plt.xticks(rotation=45, weight='bold', fontsize='medium')
plt.yticks(rotation='horizontal')
ax.set_xlabel('')
# Save figure as .svg file.
plt.savefig('./potee_methyl_percent_primary_vs_recurrent.svg', format='svg', bbox_inches='tight', pad_inches=0)
print('[Save] ./potee_methyl_percent_primary_vs_recurrent.svg \n')

print('[Begin] F')
# ************ POTEF ************
fig = plt.figure(figsize=(381.53,35))
sns.set(font_scale=12)
gs = gridspec.GridSpec(2,1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
fig.subplots_adjust(hspace=0)

# Plot heatmap with no colors (Table figure)
with sns.axes_style('white'):
    sns.heatmap(potef_fte_prim, cbar=False, square=False, xticklabels=False, yticklabels=['FTE', 'HGSC (Primary)'], annot=True, annot_kws={"weight":'bold'}, fmt='.3f', linecolor='black', linewidths=20, cmap=ListedColormap(['white']), ax=ax1)

ax = sns.heatmap(potef_fte_prim_diff.to_frame().T, cbar=False, square=False, xticklabels=True, yticklabels=['% difference'], annot=True, annot_kws={"weight":'bold'}, fmt= '.0f' ,linecolor='black', linewidths=20, cmap='OrRd_r', ax=ax2)
for t in ax.texts: t.set_text(t.get_text() + " %")
plt.xticks(rotation=45, weight='bold', fontsize='medium')
plt.yticks(rotation='horizontal')
ax.set_xlabel('')
# Save figure as .svg file.
plt.savefig('./potef_methyl_percent_primary_vs_fte.svg', format='svg', bbox_inches='tight', pad_inches=0)
print('[Save] ./potef_methyl_percent_primary_vs_fte.svg')

fig = plt.figure(figsize=(381.53,35))
sns.set(font_scale=12)
gs = gridspec.GridSpec(2,1, height_ratios=[2, 1])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
fig.subplots_adjust(hspace=0)

# Plot heatmap with no colors (Table figure)
with sns.axes_style('white'):
    sns.heatmap(potef_prim_recur_match, cbar=False, square=False, xticklabels=False, yticklabels=['HGSC (Primary)', 'HGSC (Recurrent)'], annot=True, annot_kws={"weight":'bold'}, fmt='.3f', linecolor='black', linewidths=20, cmap=ListedColormap(['white']), ax=ax1)

ax = sns.heatmap(potef_prim_recur_match_diff.to_frame().T, cbar=False, square=False, xticklabels=True, yticklabels=['% difference'], annot=True, annot_kws={"weight":'bold'}, fmt= '.0f' ,linecolor='black', linewidths=20, cmap='OrRd_r', ax=ax2)
for t in ax.texts: t.set_text(t.get_text() + " %")
plt.xticks(rotation=45, weight='bold', fontsize='medium')
plt.yticks(rotation='horizontal')
ax.set_xlabel('')
# Save figure as .svg file.
plt.savefig('./potef_methyl_percent_primary_vs_recurrent.svg', format='svg', bbox_inches='tight', pad_inches=0)
print('[Save] ./potef_methyl_percent_primary_vs_recurrent.svg \n')

print('[Done] Analysis #1 [untitled]')

%matplotlib agg
print('[Begin] Analysis #2 [untitled]')
#POTEA
print('[Run on] POTEA')
potea_meth  = compare_prim_match_to_recur_match_methyl_and_expre(case_ref, meth_POTEs_ov, specimen_ref, expre, 'POTEA', range(22,25), True)[0] # Methyl object.
potea_meth.value = potea_meth.value*100
potea_expre = compare_prim_match_to_recur_match_methyl_and_expre(case_ref, meth_POTEs_ov, specimen_ref, expre, 'POTEA', range(22,25), True)[1] # Expre object.
#POTEC
print('[Run on] POTEC')
potec_meth  = compare_prim_match_to_recur_match_methyl_and_expre(case_ref, meth_POTEs_ov, specimen_ref, expre, 'POTEC', range(6,10), True)[0] # Methyl object.
potec_meth.value = potec_meth.value*100
potec_expre = compare_prim_match_to_recur_match_methyl_and_expre(case_ref, meth_POTEs_ov, specimen_ref, expre, 'POTEC', range(6,10), True)[1] # Expre object.
#POTEE
print('[Run on] POTEE')
potee_meth  = compare_prim_match_to_recur_match_methyl_and_expre(case_ref, meth_POTEs_ov, specimen_ref, expre, 'POTEE', range(1,3), True)[0] # Methyl object.
potee_meth.value = potee_meth.value*100
potee_expre = compare_prim_match_to_recur_match_methyl_and_expre(case_ref, meth_POTEs_ov, specimen_ref, expre, 'POTEE', range(1,3), True)[1] # Expre object.
#POTEF
print('[Run on] POTEF')
potef_meth  = compare_prim_match_to_recur_match_methyl_and_expre(case_ref, meth_POTEs_ov, specimen_ref, expre, 'POTEF', range(10,14), True)[0] # Methyl object.
potef_meth.value = potef_meth.value*100
potef_expre = compare_prim_match_to_recur_match_methyl_and_expre(case_ref, meth_POTEs_ov, specimen_ref, expre, 'POTEF', range(10,14), True)[1] # Expre object.
print('\n\n')

 
print('[Visualize] Analysis #2 [untitled]')
print('[Begin] A')
ax = sns.catplot(x='value', y='Case', hue='sample', data=potea_meth, orient='h', kind='bar', legend=False)
ax.set(xlabel='% Methylation (beta value)', ylabel='', yticklabels='', title='POTEA CpG sites Methylation')
plt.axvline(potea_meth[potea_meth['sample'] == 'Primary'].mean().value, color='blue', linestyle=':')
plt.axvline(potea_meth[potea_meth['sample'] != 'Primary'].mean().value, color='orange', linestyle=':')
plt.savefig('./potea_meth.svg', format='svg')
print('[Save] ./potea_meth.svg')
ax = sns.catplot(x='value', y='Case', hue='sample', data=potea_expre, orient='h', kind='bar')
ax.set(xlabel='Read count (FPKM normalized)', ylabel='', title='POTEA mRNA Expression')
plt.axvline(potea_expre[potea_expre['sample'] == 'Primary'].mean().value, color='blue', linestyle=':')
plt.axvline(potea_expre[potea_expre['sample'] != 'Primary'].mean().value, color='orange', linestyle=':')
plt.savefig('./potea_expre.svg', format='svg')
print('[Save] ./potea_expre.svg')
potea_meth_expre = pd.concat([potea_meth,potea_expre], axis=1)
potea_meth_expre.columns = ['case', 'meth_value', 'sample_1', 'experiment', 'case', 'expre_value', 'sample_2', 'experiment']
c = sns.lmplot(x='meth_value', y='expre_value', hue='sample_1', data=potea_meth_expre[['meth_value', 'expre_value', 'sample_1']], ci=False, markers=["o","x"], palette=sns.xkcd_palette(['black']), line_kws={'linestyle':':', 'linewidth':.90})
c.set_axis_labels('Methylation intensity (beta value)', 'Read count (FPKM normalized)')
plt.savefig('./scatter_prim_vs_recur_potea.svg', format='svg')
print('[Save] ./scatter_prim_vs_recur_potea.svg')
plt.figure(figsize=(4,8))
sns.set_style('ticks')
sns.boxplot(x='sample', y='value', data=potea_meth, color='white')
plt.yticks(fontsize='x-large')
plt.ylabel('')
plt.xticks(fontsize='x-large')
plt.savefig('./box_prim_vs_recur_meth_potea.svg', format='svg')
print('[Save] ./box_prim_vs_recur_meth_potea.svg')
plt.figure(figsize=(4,8))
sns.set_style('ticks')
sns.boxplot(x='sample', y='value', data=potea_expre, color='white')
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.yticks(fontsize='x-large')
plt.ylabel('')
plt.xticks(fontsize='x-large')
plt.savefig('./box_prim_vs_recur_expre_potea.svg', format='svg')
print('[Save] ./box_prim_vs_recur_expre_potea.svg \n')

print('[Begin] C')
ax = sns.catplot(x='value', y='Case', hue='sample', data=potec_meth, orient='h', kind='bar', legend=False)
ax.set(xlabel='% Methylation (beta value)', ylabel='', yticklabels='', title='POTEC CpG sites Methylation')
plt.axvline(potec_meth[potec_meth['sample'] == 'Primary'].mean().value, color='blue', linestyle=':')
plt.axvline(potec_meth[potec_meth['sample'] != 'Primary'].mean().value, color='orange', linestyle=':')
plt.savefig('./potec_meth.svg', format='svg')
print('[Save] ./potec_meth.svg')
ax = sns.catplot(x='value', y='Case', hue='sample', data=potec_expre, orient='h', kind='bar')
ax.set(xlabel='Read count (FPKM normalized)', ylabel='', title='POTEC mRNA Expression')
plt.axvline(potec_expre[potec_expre['sample'] == 'Primary'].mean().value, color='blue', linestyle=':')
plt.axvline(potec_expre[potec_expre['sample'] != 'Primary'].mean().value, color='orange', linestyle=':')
plt.savefig('./potec_expre.svg', format='svg')
print('[Save] ./potec_expre.svg')
potec_meth_expre = pd.concat([potec_meth,potec_expre], axis=1)
potec_meth_expre.columns = ['case', 'meth_value', 'sample_1', 'experiment', 'case', 'expre_value', 'sample_2', 'experiment']
c = sns.lmplot(x='meth_value', y='expre_value', hue='sample_1', data=potec_meth_expre[['meth_value', 'expre_value', 'sample_1']], ci=False, markers=["o","x"], palette=sns.xkcd_palette(['black']), line_kws={'linestyle':':', 'linewidth':.90})
c.set_axis_labels('Methylation intensity (beta value)', 'Read count (FPKM normalized)')
plt.savefig('./scatter_prim_vs_recur_potec.svg', format='svg')
print('[Save] ./scatter_prim_vs_recur_potec.svg')
plt.figure(figsize=(4,8))
sns.set_style('ticks')
sns.boxplot(x='sample', y='value', data=potec_meth, color='white')
plt.yticks(fontsize='x-large')
plt.ylabel('')
plt.xticks(fontsize='x-large')
plt.savefig('./box_prim_vs_recur_meth_potec.svg', format='svg')
print('[Save] ./box_prim_vs_recur_meth_potec.svg')
plt.figure(figsize=(4,8))
sns.set_style('ticks')
sns.boxplot(x='sample', y='value', data=potec_expre, color='white')
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.yticks(fontsize='x-large')
plt.ylabel('')
plt.xticks(fontsize='x-large')
plt.savefig('./box_prim_vs_recur_expre_potec.svg', format='svg')
print('[Save] ./box_prim_vs_recur_expre_potec.svg \n')

print('[Begin] E')
ax = sns.catplot(x='value', y='Case', hue='sample', data=potee_meth, orient='h', kind='bar', legend=False)
ax.set(xlabel='% Methylation (beta value)', ylabel='', yticklabels='', title='POTEE CpG sites Methylation')
plt.axvline(potee_meth[potee_meth['sample'] == 'Primary'].mean().value, color='blue', linestyle=':')
plt.axvline(potee_meth[potee_meth['sample'] != 'Primary'].mean().value, color='orange', linestyle=':')
plt.savefig('./potee_meth.svg', format='svg')
print('[Save] ./potee_meth.svg')
ax = sns.catplot(x='value', y='Case', hue='sample', data=potee_expre, orient='h', kind='bar')
ax.set(xlabel='Read count (FPKM normalized)', ylabel='', title='POTEE mRNA Expression')
plt.axvline(potee_expre[potee_expre['sample'] == 'Primary'].mean().value, color='blue', linestyle=':')
plt.axvline(potee_expre[potee_expre['sample'] != 'Primary'].mean().value, color='orange', linestyle=':')
plt.savefig('./potee_expre.svg', format='svg')
print('[Save] ./potee_expre.svg')
potee_meth_expre = pd.concat([potee_meth,potee_expre], axis=1)
potee_meth_expre.columns = ['case', 'meth_value', 'sample_1', 'experiment', 'case', 'expre_value', 'sample_2', 'experiment']
c = sns.lmplot(x='meth_value', y='expre_value', hue='sample_1', data=potee_meth_expre[['meth_value', 'expre_value', 'sample_1']], ci=False, markers=["o","x"], palette=sns.xkcd_palette(['black']), line_kws={'linestyle':':', 'linewidth':.90})
c.set_axis_labels('Methylation intensity (beta value)', 'Read count (FPKM normalized)')
plt.savefig('./scatter_prim_vs_recur_potee.svg', format='svg')
print('[Save] ./scatter_prim_vs_recur_potee.svg')
plt.figure(figsize=(4,8))
sns.set_style('ticks')
sns.boxplot(x='sample', y='value', data=potee_meth, color='white')
plt.yticks(fontsize='x-large')
plt.ylabel('')
plt.xticks(fontsize='x-large')
plt.savefig('./box_prim_vs_recur_meth_potee.svg', format='svg')
print('[Save] ./box_prim_vs_recur_meth_potee.svg')
plt.figure(figsize=(4,8))
sns.set_style('ticks')
sns.boxplot(x='sample', y='value', data=potee_expre, color='white')
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.yticks(fontsize='x-large')
plt.ylabel('')
plt.xticks(fontsize='x-large')
plt.savefig('./box_prim_vs_recur_expre_potee.svg', format='svg')
print('[Save] ./box_prim_vs_recur_expre_potee.svg \n')

print('[Begin] F')
ax = sns.catplot(x='value', y='Case', hue='sample', data=potef_meth, orient='h', kind='bar', legend=False)
ax.set(xlabel='% Methylation (beta value)', ylabel='', yticklabels='', title='POTEF CpG sites Methylation')
plt.axvline(potef_meth[potef_meth['sample'] == 'Primary'].mean().value, color='blue', linestyle=':')
plt.axvline(potef_meth[potef_meth['sample'] != 'Primary'].mean().value, color='orange', linestyle=':')
plt.savefig('./potef_meth.svg', format='svg')
print('[Save] ./potef_meth.svg')
ax = sns.catplot(x='value', y='Case', hue='sample', data=potef_expre, orient='h', kind='bar')
ax.set(xlabel='Read count (FPKM normalized)', ylabel='', title='POTEF mRNA Expression')
plt.axvline(potef_expre[potef_expre['sample'] == 'Primary'].mean().value, color='blue', linestyle=':')
plt.axvline(potef_expre[potef_expre['sample'] != 'Primary'].mean().value, color='orange', linestyle=':')
plt.savefig('./potef_expre.svg', format='svg')
print('[Save] ./potef_expre.svg')
potef_meth_expre = pd.concat([potef_meth,potef_expre], axis=1)
potef_meth_expre.columns = ['case', 'meth_value', 'sample_1', 'experiment', 'case', 'expre_value', 'sample_2', 'experiment']
c = sns.lmplot(x='meth_value', y='expre_value', hue='sample_1', data=potef_meth_expre[['meth_value', 'expre_value', 'sample_1']], ci=False, markers=["o","x"], palette=sns.xkcd_palette(['black']), line_kws={'linestyle':':', 'linewidth':.90})
c.set_axis_labels('Methylation intensity (beta value)', 'Read count (FPKM normalized)')
plt.savefig('./scatter_prim_vs_recur_potef.svg', format='svg')
print('[Save] ./scatter_prim_vs_recur_potef.svg')
plt.figure(figsize=(4,8))
sns.set_style('ticks')
sns.boxplot(x='sample', y='value', data=potef_meth, color='white')
plt.yticks(fontsize='x-large')
plt.ylabel('')
plt.xticks(fontsize='x-large')
plt.savefig('./box_prim_vs_recur_meth_potef.svg', format='svg')
print('[Save] ./box_prim_vs_recur_meth_potef.svg')
plt.figure(figsize=(4,8))
sns.set_style('ticks')
sns.boxplot(x='sample', y='value', data=potef_expre, color='white')
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')
plt.yticks(fontsize='x-large')
plt.ylabel('')
plt.xticks(fontsize='x-large')
plt.savefig('./box_prim_vs_recur_expre_potef.svg', format='svg')
print('[Save] ./box_prim_vs_recur_expre_potef.svg \n')

print('[Done] Analysis #2 [untitled]')

potef_meth_mx  = methyl_expre_integrator(meth_POTEs_ov, meth_ref, expre, 'POTEF', range(10,14), True)[0] # Methyl mx.
potef_expre_mx = methyl_expre_integrator(meth_POTEs_ov, meth_ref, expre, 'POTEF', range(10,14), True)[1] # Expre mx.
potef_meth_expre_comb = pd.concat([potef_meth_mx, potef_expre_mx], axis=1)
potef_meth_expre_comb['CpG Methylation (β-value)'] = potef_meth_expre_comb.iloc[:,:-1].mean(axis=1)
potef_meth_expre_comb.rename({'normalized_read_count':'mRNA Expression (FPKM normalized)'}, axis=1, inplace=True)


potef_g = sns.clustermap(potef_meth_mx, col_cluster=False, square=True, linewidth=.1, figsize=(20,20), cmap='gray_r')
potef_cluster_order = potef_g.dendrogram_row.reordered_ind
plt.savefig('./potef_meth_clustermap.svg', format='svg')
print('[Save] ./potef_meth_clustermap.svg')
plt.figure(figsize=(20,20))
sns.heatmap(potef_expre_mx.reset_index().reindex(potef_cluster_order).set_index('ID'), square=True, linewidth=.1, cmap='RdGy_r')
plt.savefig('./potef_expre_clustermap.svg', format='svg')
print('[Save] ./potef_expre_clustermap.svg')
potef_scatter= sns.jointplot(x='CpG Methylation (β-value)', y='mRNA Expression (FPKM normalized)', data=potef_meth_expre_comb, kind='reg', color='black', ci=False)
potef_scatter.annotate(stats.pearsonr)
plt.savefig('./potef_meth_vs_expre_scatter.svg', format='svg')
print('[Save] ./potef_meth_vs_expre_scatter.svg \n')

%matplotlib agg
print('[Begin] Analysis #3 [untitled]')
#POTEA
potea_meth_mx  = methyl_expre_integrator(meth_POTEs_ov, meth_ref, expre, 'POTEA', range(22,25), True)[0] # Methyl mx.
potea_expre_mx = methyl_expre_integrator(meth_POTEs_ov, meth_ref, expre, 'POTEA', range(22,25), True)[1] # Expre mx.
potea_meth_expre_comb = pd.concat([potea_meth_mx, potea_expre_mx], axis=1)
potea_meth_expre_comb['CpG Methylation (β-value)'] = potea_meth_expre_comb.iloc[:,:-1].mean(axis=1)
potea_meth_expre_comb.rename({'normalized_read_count':'mRNA Expression (FPKM normalized)'}, axis=1, inplace=True)
# POTEC
potec_meth_mx  = methyl_expre_integrator(meth_POTEs_ov, meth_ref, expre, 'POTEC', range(6,10), True)[0] # Methyl mx.
potec_expre_mx = methyl_expre_integrator(meth_POTEs_ov, meth_ref, expre, 'POTEC', range(6,10), True)[1] # Expre mx.
potec_meth_expre_comb = pd.concat([potec_meth_mx, potec_expre_mx], axis=1)
potec_meth_expre_comb['CpG Methylation (β-value)'] = potec_meth_expre_comb.iloc[:,:-1].mean(axis=1)
potec_meth_expre_comb.rename({'normalized_read_count':'mRNA Expression (FPKM normalized)'}, axis=1, inplace=True)
# POTEE
potee_meth_mx  = methyl_expre_integrator(meth_POTEs_ov, meth_ref, expre, 'POTEE', range(1,3), True)[0] # Methyl mx.
potee_expre_mx = methyl_expre_integrator(meth_POTEs_ov, meth_ref, expre, 'POTEE', range(1,3), True)[1] # Expre mx.
potee_meth_expre_comb = pd.concat([potee_meth_mx, potee_expre_mx], axis=1)
potee_meth_expre_comb['CpG Methylation (β-value)'] = potee_meth_expre_comb.iloc[:,:-1].mean(axis=1)
potee_meth_expre_comb.rename({'normalized_read_count':'mRNA Expression (FPKM normalized)'}, axis=1, inplace=True)
# POTEF
potef_meth_mx  = methyl_expre_integrator(meth_POTEs_ov, meth_ref, expre, 'POTEF', range(10,14), True)[0] # Methyl mx.
potef_expre_mx = methyl_expre_integrator(meth_POTEs_ov, meth_ref, expre, 'POTEF', range(10,14), True)[1] # Expre mx.
potef_meth_expre_comb = pd.concat([potef_meth_mx, potef_expre_mx], axis=1)
potef_meth_expre_comb['CpG Methylation (β-value)'] = potef_meth_expre_comb.iloc[:,:-1].mean(axis=1)
potef_meth_expre_comb.rename({'normalized_read_count':'mRNA Expression (FPKM normalized)'}, axis=1, inplace=True)

print('\n\n')

 
print('[Visualize] Analysis #3 [untitled]')
print('[Begin] A')
potea_g = sns.clustermap(potea_meth_mx, col_cluster=False, square=True, linewidth=.1, figsize=(15,5.84), cmap='gray_r')
potea_cluster_order = potea_g.dendrogram_row.reordered_ind
plt.savefig('./potea_meth_clustermap.svg', format='svg')
print('[Save] ./potea_meth_clustermap.svg')
plt.figure(figsize=(15,5.84))
sns.heatmap(potea_expre_mx.reset_index().reindex(potea_cluster_order).set_index('ID'), square=True, linewidth=.1, cmap='RdGy_r')
plt.savefig('./potea_expre_clustermap.svg', format='svg')
print('[Save] ./potea_expre_clustermap.svg')
potea_scatter= sns.jointplot(x='CpG Methylation (β-value)', y='mRNA Expression (FPKM normalized)', data=potea_meth_expre_comb, kind='reg', color='black', ci=False)
potea_scatter.annotate(stats.pearsonr)
plt.savefig('./potea_meth_vs_expre_scatter.svg', format='svg')
print('[Save] ./potea_meth_vs_expre_scatter.svg \n')

print('[Begin] C')
potec_g = sns.clustermap(potec_meth_mx, col_cluster=False, square=True, linewidth=.1, figsize=(20,20), cmap='gray_r')
potec_cluster_order = potec_g.dendrogram_row.reordered_ind
plt.savefig('./potec_meth_clustermap.svg', format='svg')
print('[Save] ./potec_meth_clustermap.svg')
plt.figure(figsize=(20,20))
sns.heatmap(potec_expre_mx.reset_index().reindex(potec_cluster_order).set_index('ID'), square=True, linewidth=.1, cmap='RdGy_r')
plt.savefig('./potec_expre_clustermap.svg', format='svg')
print('[Save] ./potec_expre_clustermap.svg')
potec_scatter= sns.jointplot(x='CpG Methylation (β-value)', y='mRNA Expression (FPKM normalized)', data=potec_meth_expre_comb, kind='reg', color='black', ci=False)
potec_scatter.annotate(stats.pearsonr)
plt.savefig('./potec_meth_vs_expre_scatter.svg', format='svg')
print('[Save] ./potec_meth_vs_expre_scatter.svg \n')

print('[Begin] E')
potee_g = sns.clustermap(potee_meth_mx, col_cluster=False, square=True, linewidth=.1, figsize=(19.77,20), cmap='gray_r')
potee_cluster_order = potee_g.dendrogram_row.reordered_ind
plt.savefig('./potee_meth_clustermap.svg', format='svg')
print('[Save] ./potee_meth_clustermap.svg')
plt.figure(figsize=(19.77,20))
sns.heatmap(potee_expre_mx.reset_index().reindex(potee_cluster_order).set_index('ID'), square=True, linewidth=.1, cmap='RdGy_r')
plt.savefig('./potee_expre_clustermap.svg', format='svg')
print('[Save] ./potee_expre_clustermap.svg')
potee_scatter= sns.jointplot(x='CpG Methylation (β-value)', y='mRNA Expression (FPKM normalized)', data=potee_meth_expre_comb, kind='reg', color='black', ci=False)
potee_scatter.annotate(stats.pearsonr)
plt.savefig('./potee_meth_vs_expre_scatter.svg', format='svg')
print('[Save] ./potee_meth_vs_expre_scatter.svg \n')

print('[Begin] F')
potef_g = sns.clustermap(potef_meth_mx, col_cluster=False, square=True, linewidth=.1, figsize=(20.22,20), cmap='gray_r')
potef_cluster_order = potef_g.dendrogram_row.reordered_ind
plt.savefig('./potef_meth_clustermap.svg', format='svg')
print('[Save] ./potef_meth_clustermap.svg')
plt.figure(figsize=(20.22,20))
sns.heatmap(potef_expre_mx.reset_index().reindex(potef_cluster_order).set_index('ID'), square=True, linewidth=.1, cmap='RdGy_r')
plt.savefig('./potef_expre_clustermap.svg', format='svg')
print('[Save] ./potef_expre_clustermap.svg')
potef_scatter= sns.jointplot(x='CpG Methylation (β-value)', y='mRNA Expression (FPKM normalized)', data=potef_meth_expre_comb, kind='reg', color='black', ci=False)
potef_scatter.annotate(stats.pearsonr)
plt.savefig('./potef_meth_vs_expre_scatter.svg', format='svg')
print('[Save] ./potef_meth_vs_expre_scatter.svg \n')

print('[Done] Analysis #3 [untitled]')

print('a',potea_meth_mx.shape)
print('c',potec_meth_mx.shape)
print('e',potee_meth_mx.shape)
print('f',potef_meth_mx.shape)

