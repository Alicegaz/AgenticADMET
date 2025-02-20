import json
import random

def get_random_half_dict(original_dict):
    # Get the number of items to select (half of the dictionary length)
    num_items = len(original_dict) // 3
    
    # Convert dictionary items to a list
    items = list(original_dict.items())
    
    # Randomly select half of the items
    selected_items = random.sample(items, num_items)
    
    # Convert back to dictionary
    result_dict = dict(selected_items)
    
    return result_dict

with open("polaris-antiviral-admet-2025.json", "r") as f:
    polaris_dataset = json.load(f)

polaris_dataset_train = polaris_dataset.copy()
polaris_dataset_train.pop("CO[C@H]1C[C@H](N2N=CC3=C(C(=O)NC4=CC=C5CNCC5=C4)C=C(Cl)C=C32)C1")
polaris_dataset_train = get_random_half_dict(polaris_dataset_train)



prompt1 = f"""Background:
LogD is a measure of a molecule's lipophilicity, or its ability to dissolve in fats. It's a crucial property in drug discovery, as it influences factors like absorption, distribution, metabolism, and excretion (ADME).
Mouse Liver Microsomal stability (MLM, protocol): This is a stability assay that tests how quickly a molecule gets broken down by mouse liver microsomes. This is a useful assay that can be used as an estimate on how long a molecule will reside in the mouse body before it gets cleared.
Human Liver Microsomal stability (HLM, protocol): This is a stability assay that tests how quickly a molecule gets broken down by human liver microsomes. This is a useful assay that can be used as an estimate on how long a molecule will reside in the human body before it gets cleared.
Solubility (KSOL, protocol): solubility is essential for drug molecules: this heavily affects the pharmacokinetic and dynamics ('PKPD') of the molecule in the human body.
Cell permeation (MDR1-MDCKII, protocol): MDCKII-MDR1 is a cell line that's used to model cell permeation i.e. how well drug compounds will permeate cell layers. For coronaviruses this is a critical endpoint because there is increasing evidence that afflictions such as long-covid are caused by (remnant) virus particles in the brain, and blood-brain-barrier (BBB) permeation is critical for drug candidates to reach the brain.

Additional Context:

Here is the first batch of molecules we recieved wet lab results for:
{str(polaris_dataset_train)}

Manual Rules Used by Chemists in Drug Discovery

When evaluating a batch of molecules with ADMET data, experienced chemists use a combination of knowledge, intuition, and established rules to guide their decision-making. Here are some of the key manual rules they often employ:

Lipinski's Rule of Five (Ro5): This rule helps predict the oral bioavailability of a drug candidate. It states that a molecule is likely to have good oral absorption if it meets the following criteria:

Molecular weight ≤ 500 Da
Lipophilicity (logP) ≤ 5
Number of hydrogen bond acceptors ≤ 10
Number of hydrogen bond donors ≤ 5
Pfizer Rule: This rule focuses on potential toxicity. It suggests that compounds with high lipophilicity (logP > 3) and low polar surface area (TPSA < 75) are more likely to be toxic.

GSK Rule: This rule proposes that compounds with molecular weight ≤ 400 Da and logP ≤ 4 are more likely to have a favorable ADMET profile.

Golden Triangle Rule: This rule suggests that compounds with 200 Da ≤ molecular weight ≤ 350 Da and -2 ≤ logD ≤ 5 are more likely to have a favorable ADMET profile.

Veber's Rule : This rule expands on Lipinski's Ro5 by considering molecular flexibility. It suggests that good oral bioavailability is likely if:   

Number of rotatable bonds ≤ 10
Polar surface area ≤ 140 Å² or Total number of hydrogen bond donors and acceptors ≤ 12
Rule of Three (Ro3) : This rule is often applied in fragment-based drug discovery. It suggests that fragment hits should ideally have:   

Molecular weight ≤ 300 Da
cLogP ≤ 3
Number of hydrogen bond donors and acceptors ≤ 3
Other Considerations:

"Lead-likeness" : Similar to drug-likeness, but with more relaxed criteria, as lead compounds are often optimized further.   
Ligand Efficiency (LE) : A measure of binding affinity relative to the size of the molecule. Higher LE is generally desirable.   
Lipophilic Efficiency (LiPE) : A measure of potency relative to lipophilicity. Higher LiPE suggests better balance between these properties.   
Synthetic Accessibility : The ease with which a molecule can be synthesized. Easier synthesis is generally preferred.   
Important Notes:

These rules are guidelines, not absolute requirements. There are exceptions to every rule, and the specific context of the drug discovery program should always be considered.
Experienced chemists often develop their own intuition and heuristics based on their experience and knowledge of specific drug targets and therapeutic areas.
Computational tools and predictive models are increasingly used to complement these manual rules and provide more quantitative predictions of ADMET properties.
By combining these manual rules with their expertise and the available data, chemists can make informed decisions about which molecules to prioritize for further development and optimization.

As a cheif chemist DM, comprehensively study the results and look at a candidate molecule CO[C@H]1C[C@H](N2N=CC3=C(C(=O)NC4=CC=C5CNCC5=C4)C=C(Cl)C=C32)C1. Based on its bonds, fragments, atoms, first batch results and your chemistry knoledge, what do you think is its logD value?

Output results in one of the three ranges

LogD (Lipophilicity):
High-Risk (High Lipophilicity): LogD > 3. These molecules are highly lipophilic and may have increased metabolism, poor solubility, potential for toxicity, and higher risk of off-target binding. Modification to reduce lipophilicity is often needed.
Weakly Optimized (Low Lipophilicity): LogD < 1. These molecules are very polar and may have poor permeability. Enhancement of lipophilicity might be necessary to improve absorption and distribution.
Acceptable (Optimal Lipophilicity): 1 ≤ LogD ≤ 3. These molecules have a good balance of lipophilicity and polarity, which is generally favorable for drug-like properties.

HLM (Human Liver Microsomal Stability):
High-Risk (Rapid Metabolism): T½ < 15 minutes. These molecules are rapidly metabolized and likely to have poor oral bioavailability. Significant optimization is usually needed.
Weakly Optimized (Moderate Metabolism): 15 minutes ≤ T½ < 30 minutes. These molecules have moderate stability and may require some optimization to improve their pharmacokinetic properties.
Acceptable (Good Stability): T½ ≥ 30 minutes. These molecules exhibit good stability and are less likely to be limited by hepatic metabolism.

MLM (Mouse Liver Microsomal Stability):
High-Risk (Rapid Metabolism): T½ < 15 minutes
Weakly Optimized (Moderate Metabolism): 15 minutes ≤ T½ < 30 minutes
Acceptable (Good Stability): T½ ≥ 30 minutes

KSOL (Solubility):
High-Risk (Poor Solubility): logS < -5. These molecules have very low solubility and are likely to face significant challenges with formulation and absorption. Major optimization is often needed.
Weakly Optimized (Moderate Solubility): -5 ≤ logS < -4. These molecules have moderate solubility and may require some formulation efforts or structural modifications to improve their solubility.
Acceptable (Good Solubility): logS ≥ -4. These molecules exhibit good solubility and are less likely to be limited by solubility issues.

MDR1-MDCKII (Cell Permeability):
High-Risk (Poor Permeability): Papp < 1 x 10⁻⁶ cm/s. These molecules have very low permeability and are likely to face challenges with absorption and reaching their target. Significant optimization is usually needed.
Weakly Optimized (Moderate Permeability): 1 x 10⁻⁶ cm/s ≤ Papp < 2 x 10⁻⁶ cm/s. These molecules have moderate permeability and may benefit from some optimization to improve their absorption and distribution.
Acceptable (Good Permeability): Papp ≥ 2 x 10⁻⁶ cm/s. These molecules exhibit good permeability and are less likely to be limited by permeability issues."""