## Exploratory Data Analysis

### Dataset Overview
- **Total samples:** 5,574 SMS messages
- **Features:** 2 columns (`sms` text, `label`)
- **Data quality:** No missing values; 403 duplicate messages (7.2%) removed prior to modeling to prevent data leakage

### Class Imbalance
- **Ham:** 4,827 messages (86.6%)
- **Spam:** 747 messages (13.4%)
- **Imbalance ratio:** 6.5:1 (ham:spam)
- **Implication:** Accuracy alone is misleading on this dataset. Evaluation relies on precision, recall, and F1-score for the spam class specifically, with a stratified train/test split to preserve class ratio.

### Text Length & Structure
- **Mean length:** 81.5 characters (median: 63) — right-skewed, driven by a small number of long messages (max: 911 characters)
- **Average word count:** ~15 words per message; average unique words: ~14.5
- Most frequent overall words are stopwords (`to`, `i`, `you`, `the`), confirming the need for stopword removal before feature extraction

### Spam vs Ham — Key Differences

| Signal | Ham | Spam |
|---|---|---|
| Avg. character length | 72.5 | 139.7 |
| Avg. word count | 14.3 | 23.9 |
| Contains a URL | 0.3% | 15.9% |
| Contains a number | 15.7% | 94.8% |
| Contains "!" | 11.6% | 49.0% |
| All-caps message | 1.8% | 0.4% |

**Findings:**
- Spam messages are consistently longer and more verbose than ham (confirmed, not just a hypothesis) — length is a genuinely useful engineered feature.
- Presence of a digit is the single strongest structural signal in the dataset — nearly 95% of spam contains a number, versus 16% of ham.
- Spam vocabulary skews toward action/promotional language (`call`, `txt`, `free`, `ur`), while ham skews personal/conversational (`i`, `you`, `my`).
- All-caps usage is **not** a useful spam signal in this dataset — a counter-intuitive but real finding, worth noting since it contradicts common spam-detection assumptions.

### Cross-Validation
CV F1 scores: `[0.853, 0.896, 0.918, 0.904, 0.905]`

**Mean F1:** 0.895, **Std:** 0.022

The low standard deviation (0.022) indicates the model's performance is stable across different data splits, not a lucky fluke on one split. *5-fold cross-validation confirmed stability (mean F1 = 0.895, std = 0.022).*

### Feature Redundancy
`char_length`, `word_count`, and `unique_words` are highly correlated (r ≈ 0.97), so `char_length` was dropped to avoid redundant signal, keeping `word_count` as the representative length feature.

### Modeling Implications
- Combine TF-IDF vectorization with structural meta-features (`has_url`, `has_numbers`, `has_exclamation`, `spam_word_count`) rather than relying on word patterns alone
- Scale meta-features before combining with TF-IDF — required for distance/probability-based models to avoid one feature type dominating due to scale mismatch