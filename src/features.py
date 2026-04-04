
def engineer_features(X):
	"""
	Domain-driven feature engineering for molecular descriptor datasets.
	Computes 9 interaction features derived from toxicology and medicinal chemistry
	domain knowledge. Only applied when descriptor columns (MolLogP, qed, etc.)
	are present. For fingerprint datasets (no MolLogP column), returns X unchanged.

	New features added (descriptor datasets only):
		- herg_risk_proxy: MolLogP * fr_NH0 (hERG cardiotoxicity risk)
		- amine_tox_score: MolLogP * fr_aniline (lipophilic aromatic amine mutagenicity)
		- total_aromatic_N_burden: fr_Ar_N + fr_ArN + fr_Ar_NH (aromatic N count)
		- reactivity_spread: MaxAbsEStateIndex - MinAbsEStateIndex (reactive site spread)
		- halogen_logp_interaction: fr_halogen * MolLogP (bioaccumulation risk)
		- qed_sp3_synergy: qed * FractionCSP3 (drug-likeness + 3D complexity)
		- lipophilic_flag: (MolLogP > 3).astype(int) (Pfizer 3/75 rule, logP half)
		- aromatic_N_times_logp: total_aromatic_N_burden * MolLogP (promiscuity index)
		- allylic_ester_alert: fr_allylic_oxid + fr_ester (electrophilic alert count)

	Args:
		X (pd.DataFrame): Feature matrix. Must have a named index and named columns.
	Returns:
		pd.DataFrame: Feature matrix with new interaction columns appended.
					  Shape is (n_samples, original_cols + 9) for descriptor datasets,
					  or (n_samples, original_cols) for fingerprint datasets.
	"""
	import pandas as pd
	import numpy as np

	# Guard: only apply to descriptor datasets. Fingerprint datasets have
	# integer bit columns (0/1), not named molecular property columns.
	# MolLogP is always present in the descriptor CSV — use it as the sentinel.
	if 'MolLogP' not in X.columns:
		return X

	X = X.copy()

	# Feature 1: hERG cardiotoxicity risk proxy
	# hERG channel block requires lipophilicity + basic nitrogen. fr_NH0 counts
	# tertiary amines (the most common basic nitrogen hERG pharmacophore).
	# If fr_NH0 is missing, default to 0 (no tertiary amines detected).
	fr_NH0 = X['fr_NH0'] if 'fr_NH0' in X.columns else pd.Series(np.zeros(len(X)), index=X.index)
	X['herg_risk_proxy'] = X['MolLogP'] * fr_NH0

	# Feature 2: Aromatic amine mutagenicity amplified by lipophilicity
	# fr_aniline counts aromatic NH2 groups — a tier-1 mutagenicity alert.
	# Risk is amplified by lipophilicity because lipophilic anilines accumulate
	# and are metabolized to reactive nitroso intermediates.
	fr_aniline = X['fr_aniline'] if 'fr_aniline' in X.columns else pd.Series(np.zeros(len(X)), index=X.index)
	X['amine_tox_score'] = X['MolLogP'] * fr_aniline

	# Feature 3: Total aromatic nitrogen burden
	# Compounds with many aromatic N atoms simultaneously (pyridines, anilines,
	# aromatic secondary amines) show off-target promiscuity and CYP inhibition.
	fr_Ar_N = X['fr_Ar_N'] if 'fr_Ar_N' in X.columns else pd.Series(np.zeros(len(X)), index=X.index)
	fr_ArN = X['fr_ArN'] if 'fr_ArN' in X.columns else pd.Series(np.zeros(len(X)), index=X.index)
	fr_Ar_NH = X['fr_Ar_NH'] if 'fr_Ar_NH' in X.columns else pd.Series(np.zeros(len(X)), index=X.index)
	X['total_aromatic_N_burden'] = fr_Ar_N + fr_ArN + fr_Ar_NH

	# Feature 4: Reactivity spread (electrotopological state range)
	# Large spread between MaxAbsEStateIndex and MinAbsEStateIndex indicates
	# the molecule has both strongly electron-rich AND electron-poor regions —
	# a marker of electrophilic reactivity and covalent protein binding potential.
	if 'MaxAbsEStateIndex' in X.columns and 'MinAbsEStateIndex' in X.columns:
		X['reactivity_spread'] = X['MaxAbsEStateIndex'] - X['MinAbsEStateIndex']
	else:
		X['reactivity_spread'] = 0.0

	# Feature 5: Halogenated + lipophilic (bioaccumulation and metabolic liability)
	# Lipophilic halogens bioaccumulate and undergo reactive oxidative metabolism.
	# EPA structural alert systems flag this combination for non-genotoxic carcinogenicity.
	fr_halogen = X['fr_halogen'] if 'fr_halogen' in X.columns else pd.Series(np.zeros(len(X)), index=X.index)
	X['halogen_logp_interaction'] = fr_halogen * X['MolLogP']

	# Feature 6: Drug-likeness x 3D complexity synergy
	# qed (Quantitative Estimate of Drug-likeness) and FractionCSP3 (sp3 fraction)
	# are both individually associated with successful clinical candidates.
	# Their product captures compounds that are BOTH drug-like AND 3D-complex —
	# the opposite of flat, toxic, aromatic structures.
	if 'qed' in X.columns and 'FractionCSP3' in X.columns:
		X['qed_sp3_synergy'] = X['qed'] * X['FractionCSP3']
	else:
		X['qed_sp3_synergy'] = 0.0

	# Feature 7: Lipophilicity binary flag (Pfizer 3/75 rule — LogP half)
	# The Pfizer 3/75 rule: cLogP > 3 AND TPSA < 75 predicts in-vivo toxicity.
	# TPSA is not in the 81 selected features, so we encode the LogP threshold
	# as a binary flag so the model can combine it with PEOE_VSA descriptors
	# (which encode polar surface area information) to learn the full rule.
	X['lipophilic_flag'] = (X['MolLogP'] > 3).astype(int)

	# Feature 8: Aromatic nitrogen burden scaled by lipophilicity
	# Lipophilic multi-aromatic-nitrogen compounds are the archetypal promiscuous
	# kinase inhibitors — they generate off-target hERG, CYP, and phospholipidosis
	# liabilities simultaneously. This is a known medicinal chemistry liability space.
	X['aromatic_N_times_logp'] = X['total_aromatic_N_burden'] * X['MolLogP']

	# Feature 9: Electrophilic alert count (Michael acceptor precursors)
	# fr_allylic_oxid counts allylic oxidation sites (reactive under CYP metabolism).
	# fr_ester counts ester groups (electrophilic carbonyls susceptible to nucleophilic
	# attack by protein residues). Sum gives total count of electrophilic structural alerts.
	fr_allylic_oxid = X['fr_allylic_oxid'] if 'fr_allylic_oxid' in X.columns else pd.Series(np.zeros(len(X)), index=X.index)
	fr_ester = X['fr_ester'] if 'fr_ester' in X.columns else pd.Series(np.zeros(len(X)), index=X.index)
	X['allylic_ester_alert'] = fr_allylic_oxid + fr_ester

	return X

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

def scale_features(X, scaler_type='standard'):
	"""
	Scale features using the specified scaler.
	Args:
		X (pd.DataFrame): Feature matrix.
		scaler_type (str): 'standard', 'minmax', or 'robust'.
	Returns:
		pd.DataFrame: Scaled features.
	"""
	if scaler_type == 'standard':
		scaler = StandardScaler()
	elif scaler_type == 'minmax':
		scaler = MinMaxScaler()
	elif scaler_type == 'robust':
		scaler = RobustScaler()
	else:
		raise ValueError(f"Unknown scaler_type: {scaler_type}")
	X_scaled = scaler.fit_transform(X)
	return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

def select_features_rf(X, y, n_features=30, random_state=42):
	"""
	Select top features using Random Forest feature importances.
	Args:
		X (pd.DataFrame): Feature matrix.
		y (pd.Series): Target vector.
		n_features (int): Number of top features to select.
		random_state (int): Random seed.
	Returns:
		pd.DataFrame: Reduced feature matrix.
		list: Selected feature names.
	"""
	rf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
	rf.fit(X, y)
	importances = rf.feature_importances_
	indices = np.argsort(importances)[::-1][:n_features]
	selected_features = X.columns[indices].tolist()
	return X[selected_features], selected_features

def remove_correlated_features(X, threshold=0.9):
	"""
	Remove features with correlation above the threshold.
	Args:
		X (pd.DataFrame): Feature matrix.
		threshold (float): Correlation threshold.
	Returns:
		pd.DataFrame: Reduced feature matrix.
		list: Removed feature names.
	"""
	corr_matrix = X.corr().abs()
	upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
	to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
	return X.drop(columns=to_drop), to_drop

def pca_features(X, n_components=10, random_state=42):
	"""
	Reduce dimensionality using PCA.
	Args:
		X (pd.DataFrame): Feature matrix.
		n_components (int): Number of principal components.
		random_state (int): Random seed.
	Returns:
		pd.DataFrame: PCA-transformed features.
		PCA: Fitted PCA object.
	"""
	pca = PCA(n_components=n_components, random_state=random_state)
	X_pca = pca.fit_transform(X)
	columns = [f'PC{i+1}' for i in range(n_components)]
	return pd.DataFrame(X_pca, columns=columns, index=X.index), pca
