import pandas as pd
import numpy as np
import pickle
import os

class DataPreprocessor:
    """Pipeline de preprocessing avec sauvegarde des règles."""
    
    def __init__(self, mode='learn'):
        self.mode = mode
        self.rules = {}
        self.rules_file = 'preprocessing_rules.pkl'
        
    def load_data(self):
        prefix = 'learn' if self.mode == 'learn' else 'test'
        
        self.df = pd.read_csv(f"{prefix}_dataset.csv")
        self.df_sport = pd.read_csv(f"{prefix}_dataset_sport.csv")
        self.df_tenure = pd.read_csv(f"{prefix}_dataset_employement_tenure.csv")
        self.df_retired_former = pd.read_csv(f"{prefix}_dataset_retired_former.csv")
        self.df_job = pd.read_csv(f"{prefix}_dataset_job.csv")
        self.df_pension = pd.read_csv(f"{prefix}_dataset_retired_pension.csv")
        self.df_retired_jobs = pd.read_csv(f"{prefix}_dataset_retired_jobs.csv")
        
        self.df_code = pd.read_csv('code_CLUB.csv')
        self.df_map = pd.read_csv('code_work_desc_map.csv')[['N3', 'N2']]
        self.df_city_adm = pd.read_csv('city_adm.csv', dtype={'Insee': str, 'dep': str}, 
                                       usecols=['Insee', 'dep', 'MUNICIPALITY_TYPE'])
        self.df_departments = pd.read_csv('departments.csv', dtype={'dep': str, 'REG': str}, 
                                         usecols=['dep', 'REG'])
        self.df_city_pop = pd.read_csv('city_pop.csv', dtype={'Insee': str}, 
                                       usecols=['Insee', 'Residents'])
        self.df_loc = pd.read_csv('city_loc.csv', dtype={'Insee': str}, 
                                 usecols=['Insee', 'X', 'Y'])
        self.df_dvf = pd.read_csv('dvf2018.csv', dtype={'INSEE_COM': str})
        
        try:
            self.df_density = pd.read_csv('city_density.csv', sep=';', header=2, 
                                         dtype={'Code': str}, encoding='utf-8')
        except UnicodeDecodeError:
            self.df_density = pd.read_csv('city_density.csv', sep=';', header=2, 
                                         dtype={'Code': str}, encoding='latin-1')
        
        self.df_revenue = pd.read_csv('med_earnings_city.csv', sep=';', 
                                     dtype={'Code géographique': str})
        
    def process_sport(self):
        df_merged = self.df_sport.merge(self.df_code, left_on='CLUB', right_on='Code', how='left')
        
        if self.mode == 'learn':
            club_counts = self.df_sport["CLUB"].value_counts()
            self.rules['big_clubs'] = club_counts[club_counts > 50].index.tolist()
        
        big_clubs = self.rules['big_clubs']
        df_merged['sport_categorie'] = np.where(
            df_merged['CLUB'].isin(big_clubs),
            df_merged['CLUB'],
            df_merged['Categorie']
        )
        
        df_sport_clean = df_merged[['UNIQUE_ID', 'sport_categorie']].copy()
        self.df = self.df.merge(df_sport_clean, on='UNIQUE_ID', how='left')
        self.df['sport_categorie'] = self.df['sport_categorie'].fillna('No_Sport')
        
    def process_tenure_retired(self):
        self.df = self.df.merge(self.df_tenure, on='UNIQUE_ID', how='left')
        self.df = self.df.merge(self.df_retired_former, on='UNIQUE_ID', how='left')
        
        if 'former_employement_tenure' in self.df.columns:
            self.df['former_employement_tenure'] = self.df['former_employement_tenure'].fillna('Not_Retired')
        if 'retirement_age' in self.df.columns:
            self.df['retirement_age'] = self.df['retirement_age'].fillna(0)
        if 'former_job_42' in self.df.columns:
            self.df['former_job_42'] = self.df['former_job_42'].fillna('Not_Retired')
        if 'employement_tenure' in self.df.columns:
            self.df['employement_tenure'] = self.df['employement_tenure'].fillna('No_Job')
            
    def process_job(self):
        df_job_mapped = self.df_job.merge(self.df_map, left_on='work_desc', right_on='N3', how='left')
        df_job_mapped = df_job_mapped.drop(columns=['N3'])
        
        for col in ['employer_type', 'employee_count']:
            if col in df_job_mapped.columns:
                df_job_mapped[col] = df_job_mapped[col].fillna('NON_Observed')
        
        self.df = self.df.merge(df_job_mapped, on='UNIQUE_ID', how='left')
        
    def process_insee(self):
        self.df = self.df.merge(self.df_city_adm, on='Insee', how='left')
        self.df = self.df.merge(self.df_departments, on='dep', how='left')
        
        if self.mode == 'learn':
            villes_50 = self.df['Insee'].value_counts()[lambda x: x >= 50].index.tolist()
            self.rules['big_cities'] = villes_50
        
        big_cities = self.rules['big_cities']
        self.df['Big_City'] = self.df['Insee'].where(
            self.df['Insee'].isin(big_cities), 'Other'
        ).astype('category')
        
        self.df = self.df.merge(self.df_city_pop, on='Insee', how='left')
        self.df = self.df.merge(self.df_loc, on='Insee', how='left')
        
    def process_density_revenue(self):
        self.df['Insee'] = self.df['Insee'].astype(str).str.strip().str.zfill(5)
        self.df['Insee_Join'] = self.df['Insee'].apply(
            lambda x: '75056' if x.startswith('751') else (
                '69123' if x.startswith('693') else (
                    '13055' if x.startswith('132') else x
                )
            )
        )
        
        col_dens = [c for c in self.df_density.columns if 'population' in c and '2022' in c][0]
        dd = self.df_density.rename(columns={'Code': 'Insee_Join', col_dens: 'Densite'})[['Insee_Join', 'Densite']]
        dd['Densite'] = pd.to_numeric(dd['Densite'].astype(str).str.replace(',', '.'), errors='coerce')
        self.df = self.df.merge(dd, on='Insee_Join', how='left')
        
        dr = self.df_revenue.rename(columns={
            'Code géographique': 'Insee_Join', 
            '[DISP] Médiane (€)': 'Revenu_Median'
        })[['Insee_Join', 'Revenu_Median']]
        dr['Revenu_Median'] = pd.to_numeric(dr['Revenu_Median'], errors='coerce')
        self.df = self.df.merge(dr, on='Insee_Join', how='left')
        
        self.df.drop(columns=['Insee_Join'], inplace=True, errors='ignore')
        
    def process_pension_retired_jobs(self):
        self.df = self.df.merge(self.df_pension, on='UNIQUE_ID', how='left', suffixes=('', '_pension'))
        
        df_ret = self.df_retired_jobs.merge(self.df_map, left_on='work_desc', right_on='N3', how='left')
        df_ret = df_ret.drop(columns=['N3'], errors='ignore')
        
        if self.mode == 'learn':
            if 'N2' in df_ret.columns:
                self.rules['retired_wh_by_N2'] = df_ret.groupby('N2')['working_hours'].median().to_dict()
            else:
                self.rules['retired_wh_by_N2'] = {}
        
        if 'N2' in df_ret.columns and self.rules['retired_wh_by_N2']:
            df_ret['working_hours'] = df_ret.apply(
                lambda row: self.rules['retired_wh_by_N2'].get(row['N2'], row['working_hours']) 
                if pd.isna(row['working_hours']) else row['working_hours'],
                axis=1
            )
        df_ret['working_hours'] = df_ret['working_hours'].fillna(35.0)
        
        for col in ['employee_count', 'employer_type', 'ECO_SECT']:
            if col in df_ret.columns:
                df_ret[col] = df_ret[col].fillna('Unknown')
        
        rename_dict = {
            'N2': 'former_job_42',
            'working_hours': 'former_working_hours',
            'Type_of_contract': 'former_Type_of_contract',
            'employer_type': 'former_employer_type',
            'job_condition': 'former_job_condition',
            'employee_count': 'former_employee_count',
            'ECO_SECT': 'former_ECO_SECT',
            'work_desc': 'former_work_desc',
            'OCCUPATIONAL_STATUS': 'former_OCCUPATIONAL_STATUS',
            'Job_dep': 'former_Job_dep'
        }
        df_ret.rename(columns=rename_dict, inplace=True)
        df_ret.drop(columns='former_job_42', inplace=True, errors='ignore')
        
        self.df = self.df.merge(df_ret, on='UNIQUE_ID', how='left')
        
    def process_dvf(self):
        df_dvf = self.df_dvf.copy()
        df_dvf.rename(columns={'INSEE_COM': 'Insee'}, inplace=True)
        df_dvf['Insee'] = df_dvf['Insee'].astype(str).apply(
            lambda x: '75056' if x.startswith('751') else (
                '69123' if x.startswith('693') else (
                    '13055' if x.startswith('132') else x
                )
            )
        )
        df_dvf = df_dvf.groupby('Insee', as_index=False).mean()
        
        self.df['Insee_Join'] = self.df['Insee'].astype(str).apply(
            lambda x: '75056' if x.startswith('751') else (
                '69123' if x.startswith('693') else (
                    '13055' if x.startswith('132') else x
                )
            )
        )
        
        cols_dvf = ['Prixm2Moyen']
        self.df = self.df.merge(
            df_dvf[['Insee'] + cols_dvf], 
            left_on='Insee_Join', 
            right_on='Insee', 
            how='left', 
            suffixes=('', '_dvf')
        )
        
        self.df.drop(columns=['Insee_Join', 'Insee_dvf'], inplace=True, errors='ignore')
        
    def impute_working_hours(self):
        ids_workers = self.df_job['UNIQUE_ID'].unique()
        mask = self.df['UNIQUE_ID'].isin(ids_workers)
        
        if self.mode == 'learn':
            self.rules['wh_medians'] = {
                'by_metier_contrat_condition': self.df.loc[mask].groupby(
                    ['N2', 'Type_of_contract', 'job_condition']
                )['working_hours'].median().to_dict(),
                'by_metier_contrat': self.df.loc[mask].groupby(
                    ['N2', 'Type_of_contract']
                )['working_hours'].median().to_dict(),
                'by_metier': self.df.loc[mask].groupby('N2')['working_hours'].median().to_dict()
            }
        
        wh = self.rules['wh_medians']
        
        self.df.loc[mask, 'working_hours'] = self.df.loc[mask].apply(
            lambda row: wh['by_metier_contrat_condition'].get(
                (row['N2'], row['Type_of_contract'], row['job_condition']), 
                row['working_hours']
            ) if pd.isna(row['working_hours']) else row['working_hours'],
            axis=1
        )
        
        self.df.loc[mask, 'working_hours'] = self.df.loc[mask].apply(
            lambda row: wh['by_metier_contrat'].get(
                (row['N2'], row['Type_of_contract']), 
                row['working_hours']
            ) if pd.isna(row['working_hours']) else row['working_hours'],
            axis=1
        )
        
        self.df.loc[mask, 'working_hours'] = self.df.loc[mask].apply(
            lambda row: wh['by_metier'].get(row['N2'], row['working_hours']) 
            if pd.isna(row['working_hours']) else row['working_hours'],
            axis=1
        )
        
        self.df.loc[mask, 'working_hours'] = self.df.loc[mask, 'working_hours'].fillna(35.0)
        
    def impute_independants(self):
        mask_indep = (self.df['employement_tenure'] != 'No_Job') & (self.df['working_hours'].isna())
        
        if mask_indep.sum() == 0:
            return
        
        job_cat_cols = ['Type_of_contract', 'employer_type', 'job_condition', 
                       'work_desc', 'OCCUPATIONAL_STATUS']
        self.df.loc[mask_indep, job_cat_cols] = 'Independant'
        self.df.loc[mask_indep, 'employee_count'] = 0
        self.df.loc[mask_indep, 'Job_dep'] = self.df.loc[mask_indep, 'dep']
        
        if self.mode == 'learn':
            df_source = self.df[~mask_indep]
            self.rules['indep_medians'] = {
                'earnings_by_job': df_source.groupby('job_42')['earnings'].median().to_dict(),
                'earnings_global': df_source['earnings'].median(),
                'hours_by_job': df_source.groupby('job_42')['working_hours'].median().to_dict(),
                'hours_global': df_source['working_hours'].median()
            }
        
        medians = self.rules['indep_medians']
        
        self.df.loc[mask_indep, 'earnings'] = self.df.loc[mask_indep, 'job_42'].map(
            medians['earnings_by_job']
        ).fillna(medians['earnings_global'])
        
        self.df.loc[mask_indep, 'working_hours'] = self.df.loc[mask_indep, 'job_42'].map(
            medians['hours_by_job']
        ).fillna(medians['hours_global'])
        
    def patch_final(self):
        if 'former_dep' in self.df.columns and 'former_Job_dep' in self.df.columns:
            self.df['former_dep'] = self.df['former_dep'].fillna(self.df['former_Job_dep'])
            self.df['former_Job_dep'] = self.df['former_Job_dep'].fillna(self.df['former_dep'])
        
        if 'Retirement_benefits' in self.df.columns and 'dep' in self.df.columns:
            mask_pension = self.df['Retirement_benefits'].fillna(0) > 0
            for col in ['former_dep', 'former_Job_dep']:
                if col in self.df.columns:
                    mask_target = mask_pension & self.df[col].isna()
                    self.df.loc[mask_target, col] = self.df.loc[mask_target, 'dep']
        
        if 'earnings' in self.df.columns and 'Job_dep' in self.df.columns:
            mask_worker = (self.df['earnings'].notna()) & (self.df['Job_dep'].isna())
            self.df.loc[mask_worker, 'Job_dep'] = self.df.loc[mask_worker, 'dep']
        
        if self.mode == 'learn':
            med_age = self.df.loc[self.df['retirement_age'] > 1, 'retirement_age'].median()
            self.rules['retirement_age_median'] = med_age if not pd.isna(med_age) else 62
        
        if 'retirement_age' in self.df.columns and 'Age_2018' in self.df.columns:
            med_age = self.rules['retirement_age_median']
            mask_fix = (self.df['Act'] == 'at21') & (self.df['Age_2018'] > 62) & (self.df['retirement_age'].fillna(0) <= 1)
            self.df.loc[mask_fix, 'retirement_age'] = med_age
        
    def clean_rare_categories(self):
        THRESHOLD = 20
        cols_to_check = ['former_OCCUPATIONAL_STATUS']
        
        for col in cols_to_check:
            if col not in self.df.columns:
                continue
                
            if self.mode == 'learn':
                counts = self.df[col].value_counts()
                mode_val = counts.index[0]
                rare_cats = counts[counts < THRESHOLD].index.tolist()
                total_rare = self.df[col].isin(rare_cats).sum()
                
                self.rules[f'rare_{col}'] = {
                    'rare_cats': rare_cats,
                    'mode': mode_val,
                    'total_rare': total_rare,
                    'action': 'Other' if total_rare >= THRESHOLD else mode_val
                }
            
            rule = self.rules[f'rare_{col}']
            replacement = 'Other' if rule['action'] == 'Other' else rule['mode']
            self.df.loc[self.df[col].isin(rule['rare_cats']), col] = replacement
        
    def save_rules(self):
        if self.mode == 'learn':
            with open(self.rules_file, 'wb') as f:
                pickle.dump(self.rules, f)
        
    def load_rules(self):
        if self.mode == 'test':
            if not os.path.exists(self.rules_file):
                raise FileNotFoundError(
                    f"❌ Fichier {self.rules_file} introuvable. "
                    "Exécutez d'abord en mode 'learn'."
                )
            with open(self.rules_file, 'rb') as f:
                self.rules = pickle.load(f)
        
    def save_processed_data(self):
        if self.mode == 'learn':
            output_file = "train_processed.csv"
        else:
            output_file = "test_processed.csv"
        self.df.to_csv(output_file, index=False)
        print(f"✅ Dataset sauvegardé: {output_file}")
        
    def run_pipeline(self):
        print(f"\n🚀 PREPROCESSING START - MODE: {self.mode.upper()}")
        
        if self.mode == 'test':
            self.load_rules()
        
        self.load_data()
        self.process_sport()
        self.process_tenure_retired()
        self.process_job()
        self.process_insee()
        self.process_density_revenue()
        self.process_pension_retired_jobs()
        self.process_dvf()
        self.impute_working_hours()
        self.impute_independants()
        self.patch_final()
        self.clean_rare_categories()
        
        if self.mode == 'learn':
            self.save_rules()
        self.save_processed_data()
        
        print(f"✅ PREPROCESSING DONE - {self.df.shape[0]} lignes, {self.df.shape[1]} colonnes\n")

if __name__ == "__main__":
    preprocessor = DataPreprocessor(mode='learn')
    preprocessor.run_pipeline()