import pandas as pd
from tqdm import tqdm
import scipy.stats as stats                # for fisher test
import statsmodels.stats.multitest as smt  # for pvalue adjustment

def GO_enrichment(com2genes, GO2genes, gene2GOs, n):
    '''
    :param com2genes: dictionary where keys are community ids and values are list of genes belonging to the corresponding community
    :param GO2genes: dictionary where keys are GO terms and values are list of genes with the corresponding GO annotation
    :param gene2GOs: dictionary where keys are genes and values are list of GO terms annotated to the corresponding gene
    :param n: total number of genes in the background set
    '''
    
    nbr_coms = len(com2genes)

    # n = len(gene2GOs)
    # iff the genes without annotations belong to the dictionary with an empty list as value
#      
    ans = []
    for m in tqdm(range(nbr_coms)):
        nbr_genes_in_module = len(com2genes[m])

        # GO terms present in the module
        module_GOs = set()
        for gene in com2genes[m]:
            if gene in gene2GOs:
                module_GOs = module_GOs.union(gene2GOs[gene])            

        for go in module_GOs:
            nbr_genes_in_GO = len(GO2genes[go])

            a = len(set(GO2genes[go]).intersection(com2genes[m]))
            b = nbr_genes_in_GO - a
            c = nbr_genes_in_module - a
            d = n - nbr_genes_in_GO - c     # n - nbr_genes_in_module - b

            _, pvalue = stats.fisher_exact([[a,b],[c,d]])

            # mod, go, pval
            ans.append([m,go,pvalue])
    
    df = pd.DataFrame(ans, columns=['module','GO_id','pvalue'])

    return df


def FDR(df):
    '''
    :param df: DataFrame from GO_enrichment function
    '''
    fdr = []
    nbr_coms = df.module.nunique()
    for m in tqdm(range(nbr_coms)):
        try:
            pval_adj = smt.multipletests(pvals = df[df.module==m].pvalue, method = 'fdr_bh' )[1]
            fdr = fdr + list(pval_adj)
        except:
            fdr = fdr + df[df.module==m].pvalue.tolist()

    df['fdr'] = fdr

    return df

df = GO_enrichment(com2genes, GO2genes, gene2GOs, n)
dfa = FDR(df)

# total enriched modules
df[df.fdr < 0.05].module.nunique()
