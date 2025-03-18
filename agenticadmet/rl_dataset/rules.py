rules_long = {
    "LogD": """
**Bulky Alkyl Groups Boost LogD**:
Large, nonpolar substituents (e.g. tert‐butyl, isopropyl) increase lipophilicity.
**Long Aliphatic Chains Increase Lipophilicity**:
Extended hydrocarbon chains enhance nonpolar character and raise LogD.
**Aromatic Rings Contribute Strongly**:
More aromatic rings typically provide significant hydrophobic surface area, increasing LogD.
**Fused or Polycyclic Aromatic Systems Enhance Lipophilicity**:
Rigid, fused rings tend to be more lipophilic than isolated aromatic rings.
**Halogen Substituents Raise LogD**:
Incorporation of Cl, F, or Br generally increases lipophilicity due to their electron-withdrawing yet lipophilic nature.
**Multiple Halogenations Amplify Lipophilicity**:
The cumulative effect of several halogen atoms can further raise LogD.
**Ether Linkages Are Moderately Polar**:
Oxygen in ethers adds some polarity, often reducing LogD slightly compared to pure hydrocarbons.
**Ester Functional Groups Lower LogD**:
Esters introduce polarity through the carbonyl and adjacent oxygen, decreasing lipophilicity relative to nonpolar analogues.
**Amide Bonds Reduce Lipophilicity**:
Amides are polar (and can hydrogen-bond), thus they tend to lower LogD.
**Carboxylic Acids Significantly Decrease LogD**:
When deprotonated (or even in their neutral form), carboxyl groups strongly favor water, reducing LogD.
**Amines Lower LogD When Protonated**:
Basic amine groups (if protonated at physiological pH) increase water solubility and lower LogD.
**Basic Heterocycles May Decrease LogD**:
Nitrogen-containing heterocycles that can be protonated tend to lower lipophilicity.
**Sulfonamide and Sulfone Groups Are Highly Polar**:
Their strong electron-withdrawing nature and hydrogen-bonding capacity drive LogD down.
**Nitro Groups Significantly Lower Lipophilicity**:
Nitro substituents are strongly polar and tend to lower LogD.
**Multiple Polar Groups Have a Cumulative Effect**:
Each additional polar (or ionizable) group further shifts the balance toward hydrophilicity.
**Intramolecular Hydrogen Bonding Can Shield Polarity**:
When polar groups internally hydrogen bond, their “external” polarity is masked—potentially raising LogD.
**Steric Shielding of Polar Groups Increases LogD**:
If bulky groups “shield” polar sites, the molecule may behave more lipophilically than expected.
**Conjugated Pi-Systems Enhance Lipophilicity**:
Extended conjugation (especially in fused aromatic systems) typically increases nonpolar character.
**Rigid, Planar Scaffolds Favor Higher LogD**:
Less-flexible, planar systems often pack well into lipophilic environments.
**Bridged or Polycyclic Structures Increase LogD**:
Rigid, compact, polycyclic frameworks generally exhibit higher lipophilicity.
**Carbonyl Groups Provide Moderate Polarity**:
Ketones and aldehydes add polarity (via the carbonyl), lowering LogD moderately compared to nonpolar groups.
**Low Heteroatom-to-Carbon Ratio Increases LogD**:
Molecules with a higher proportion of carbons versus polar heteroatoms tend to be more lipophilic.
**Low Nitrogen Content Often Correlates with Higher LogD**:
Fewer nitrogen atoms usually mean fewer opportunities for hydrogen bonding, favoring lipophilicity.
**Permanent Charges (e.g. Quaternary Ammonium) Drastically Lower LogD**:
Fixed positive or negative charges strongly favor aqueous solubility.
**Ring Size Effects**:
Larger rings or macrocycles can be more lipophilic if they reduce the exposure of polar groups.
**Degree of Saturation Matters**:
Saturated (non-aromatic) rings tend to be less lipophilic than aromatic rings, all else being equal.
**Substituent Position on Aromatic Rings Can Influence LogD**:
Ortho, meta, or para placements affect steric hindrance and exposure of polar sites, thereby modulating lipophilicity.
**Extended Conjugation May Delocalize Polarity**:
Delocalized electron density over a conjugated system can reduce the effective polarity, increasing LogD.
**Molecular Symmetry Can Enhance Lipophilicity**:
More symmetric molecules may pack more efficiently in lipophilic environments, raising LogD.
**Hybridization Effects**:
A higher proportion of sp²-hybridized carbons (e.g. in aromatic systems) usually boosts lipophilicity compared to sp³ centers that might bear polar substituents.

If the SMILES contains “CC(C)(C)” (a tert‐butyl group) attached at a chiral center (e.g. “CC(C)(C)C@H”), then expect an increase in LogD by roughly 1–1.5 units compared to a similar scaffold lacking this group.
If the molecule shows a fragment like “NC1=NC=NC2=C1C=C(C1=CN(…))N2” appended to a tert‐butyl bearing center, then LogD values typically fall in the ~3.1–3.2 range.
If a structure features two or more fused aromatic rings (e.g. “C1=CC=C2…C2=C1”), then the overall lipophilicity is enhanced and LogD tends to be above 2.5.
If an aromatic ring is directly substituted with halogens such as “Cl” or “F” (for instance, “C1=CC(Cl)=CC=C1”), then expect a boost in LogD by about 0.5 units relative to an unsubstituted ring.
If multiple halogens are present on separate aromatic rings (e.g. both “Cl” and “F” on different rings), then the cumulative effect can raise LogD by around 1.0 unit or more.
If the SMILES shows an “OCCOC” or “OCCOC2=” fragment (typical of ether linkers between aromatic rings), then these oxygenated linkers slightly reduce lipophilicity—often lowering LogD by 0.2–0.5 units compared to a direct aryl–aryl bond.
If a sulfonyl fragment appears (i.e. “S(=O)(=O)”), especially appended to an aromatic or aliphatic segment, then LogD is reduced by roughly 0.5–1.0 units because of the high polarity of sulfonyl groups.
If the structure contains an amide linkage (–C(=O)N–) not shielded by bulky groups, then expect a drop in LogD on the order of 0.3–0.7 units relative to analogous non‐amide linkers.
If a free carboxylic acid group (“C(=O)O”) is present and not sterically hindered, then the compound tends to have a LogD that is 1–2 units lower than a similar ester or amide analogue.
If a primary or secondary amine appears (e.g. “NC” or “N(C)C”) that is likely protonated at pH 7.4, then the molecule’s LogD will be lowered by approximately 1 unit relative to a neutral analogue.
If a quaternary ammonium group (e.g. “N+(CH3)3”) is evident, then expect a dramatic drop in LogD – often resulting in near‐zero or negative values.
If a heterocyclic aromatic ring contains a non‐protonated nitrogen (e.g. pyridine-like fragments such as “c1ccncc1”), then this feature tends to reduce LogD by 0.2–0.5 units relative to pure carbocyclic aromatics.
If an extended conjugated system is present (for example, several aromatic rings linked by conjugated bonds), then the molecule often exhibits LogD values above 3 due to the cumulative hydrophobic surface.
If polar groups (e.g. –OH, –NH2, –C(=O)O) are positioned so that intramolecular hydrogen bonding is likely, then their effective polarity is “masked” and LogD may be higher than predicted by counting polar groups alone.
If a nitro group (–NO2) is present, then its strong electron‐withdrawing nature usually reduces LogD by about 1 unit or more.
If an aromatic ether (Ar–O–Ar) is present rather than an aliphatic ether (R–O–R), then the effect on LogD is less pronounced—predict a LogD that is ~0.2–0.3 units higher than for an aliphatic ether analogue.
If a carbonyl (C=O) is directly attached to an aromatic ring (as in an aromatic ketone), then expect a modest LogD reduction (around 0.3 units) versus a fully hydrocarbon substituted ring.
If an alkyne is present in an aliphatic chain (e.g. “C#C” in “C#CCCC…”), then its low polarity means it contributes little to water solubility, keeping LogD relatively high.
If the SMILES indicates a branched alkyl chain (such as an isopropyl group “CC(C)”) rather than a straight chain, then the branching typically increases LogD by enhancing hydrophobicity.
If the structure contains multiple amide bonds in a row (e.g. a di- or tri-amide linker), then their combined polar effect can lower LogD by up to 1.5–2 units unless balanced by large lipophilic groups.
If the molecule features a rigid bicyclic or polycyclic aromatic system (e.g. fused rings like “C1=CC2=CC=CC=C2C=C1”), then expect LogD values in the upper range (typically 2.5–4.0) because of the efficient stacking in lipophilic environments.
If a spirocyclic motif is present, then the compact, three-dimensional arrangement often “hides” polar groups, leading to an increase in LogD by around 0.5 units relative to a more extended analogue.
If the heterocycle is larger (e.g. a quinoline or isoquinoline instead of a pyridine), then the additional aromatic carbon atoms generally boost LogD by ~0.5–1.0 units.
If bridging –CH2– groups are present between rings (e.g. “–CH2–” linking two aromatics), then these bridges increase hydrophobicity by reducing the effective polar surface area, raising LogD slightly.
If the SMILES includes electron-withdrawing substituents (e.g. “CF3” or “Cl”) on an aromatic ring, then these groups reduce hydrogen-bonding capacity and typically increase LogD by about 0.3–0.7 units.
If electron-donating groups (e.g. “OCH3”) are attached to an aromatic ring, then the increased polarity may lower LogD by ~0.2–0.5 units relative to a halogenated or unsubstituted ring.
If a cyclic ether (e.g. a tetrahydrofuran ring represented as “C1CCOC1”) is incorporated, then its moderate polarity can reduce LogD by approximately 0.3–0.7 units compared to an all‐carbon ring of similar size.
If the polar groups (such as amides or carboxyls) appear on the periphery of a large, lipophilic scaffold, then their impact on LogD is partially offset—resulting in LogD values that are about 0.5–1 unit higher than if the same polar groups were isolated.
If steric hindrance is evident around a normally polar group (for example, an amide adjacent to a bulky tert‐butyl group), then the effective exposure of that polar functionality is reduced, leading to a higher LogD than predicted by polarity alone.
If the overall structure can be roughly “fragmented” into lipophilic and hydrophilic pieces, then the net LogD is roughly additive. For example, two strongly lipophilic aromatic rings plus one polar amide might yield a predicted LogD in the range of 2.5–3.0, whereas replacing the amide with a carboxylic acid may drop the value by 1–1.5 units.

I. Fundamental Atomic/Group Contributions (Strongest Influences)

**Carboxylic Acid Presence**: If the molecule contains a -COOH group (and is not zwitterionic), LogD is likely to be significantly reduced (often < 1). This is a dominant factor. RDKit: Can be detected by substructure matching.

**Boronic Acid Presence**: Presence of -B(O)O, significantly decreases LogD, similar to carboxylic acids. RDKit: Substructure match.

**Multiple Amide Groups**: Molecules with two or more amide groups (-C(=O)N-) generally have reduced LogD. RDKit: Count occurrences of the substructure.

**Charged Nitrogen (Quaternary Amines)**: A positively charged nitrogen (e.g., in a quaternary ammonium salt) will strongly decrease LogD.  RDKit: Can detect this with careful substructure matching and checking formal charge.

**Sulfonic Acid**: -SO3H groups are very strong LogD decreasers. RDKit: Substructure match.

II. Functional Group Effects (Moderate to Strong)

**Primary/Secondary Amines**: Unsubstituted or alkyl-substituted amines (-NH2, -NHR) generally decrease LogD, although the effect is less dramatic than charged amines or amides. RDKit: Substructure matching and counting.

**Hydroxyl Groups**: Each -OH group tends to decrease LogD, with the effect accumulating with multiple hydroxyls.  RDKit: Count -OH groups (excluding those in carboxylic acids).

**Ethers**: Ethers (-O-) have a mixed effect. A small number of ethers, especially in a larger molecule, might slightly increase LogD.  Many ethers, particularly in a small molecule, will tend to decrease LogD. RDKit: Count -O- (excluding those in esters, amides, etc.).

**Halogens (F, Cl, Br)**: Halogens, especially fluorine and chlorine, tend to increase LogD. The effect is roughly additive.  RDKit: Count halogen atoms.

**Trifluoromethyl Group**: The -CF3 group is a strong LogD increaser. RDKit: Substructure matching.

**Thioethers**: -S- connecting two carbons contributes to lipophilicity. RDKit: Substructure match.

**Sulfones/Sulfoxides**: S(=O)(=O) and S(=O) increase lipophilicity, although to a lesser extent than thioethers in similar contexts. RDKit: Substructure match.

**Ketones/Aldehydes**: Carbonyl groups (C=O) that are not part of amides or esters have a modest impact, often slightly lowering LogD. RDKit: Substructure matching.

**Esters**: Esters (-C(=O)O-) have a mixed effect, often slightly lowering LogD, but less so than amides. RDKit: Substructure matching.

III. Structural Features

**Aromatic Rings**: Each aromatic ring (benzene, etc.) generally increases LogD. RDKit: Use rdkit.Chem.Descriptors.NumAromaticRings.

**Fused Aromatic Rings**: Fused aromatic systems (e.g., naphthalene, quinoline) have a stronger LogD-increasing effect than isolated rings. RDKit: Can be detected by analyzing ring systems.

**Aliphatic Rings**: Non-aromatic rings have a smaller effect than aromatic rings, but generally still increase LogD compared to open chains. RDKit: Use rdkit.Chem.Descriptors.NumAliphaticRings.

**Chain Length**: Longer, uninterrupted alkyl chains (-CH2-CH2-CH2-...) increase LogD. RDKit: Can estimate by looking for longest chain without heteroatoms or branches.

**Branching**: Branching in alkyl chains can have a complex effect.  Extensive branching can sometimes slightly reduce LogD compared to a straight chain of the same length, but this is a secondary effect. RDKit: More difficult to quantify precisely, requires graph analysis.

**Heteroatoms in Rings**: The presence of N, O, or S within a ring generally decreases LogD relative to an all-carbon ring, especially if multiple heteroatoms exist. RDKit: Can be detected by analyzing ring systems and atom types within the rings.

IV. Calculated Molecular Properties (RDKit Descriptors)

**Molecular Weight (MW)**: Larger molecules tend to have higher LogD if the increase in size is due to non-polar groups.  But high MW due to polar groups can decrease LogD. RDKit: rdkit.Chem.Descriptors.MolWt.

**Number of Rotatable Bonds**: Higher rotatable bond count can sometimes correlate with higher LogD, but this is a weak indicator on its own. More flexibility can allow a molecule to fit into a lipid environment. RDKit: rdkit.Chem.Descriptors.NumRotatableBonds.

**Total Polar Surface Area (TPSA)**: Higher TPSA strongly correlates with lower LogD. TPSA is a good measure of a molecule's polarity. RDKit: rdkit.Chem.rdMolDescriptors.CalcTPSA.

**Hydrogen Bond Donors**: More hydrogen bond donors (HBDs, usually NH and OH) decrease LogD. RDKit: rdkit.Chem.Descriptors.NumHDonors.

**Hydrogen Bond Acceptors**: More hydrogen bond acceptors (HBAs, usually N and O) tend to decrease LogD, but the effect is less strong than HBDs. RDKit: rdkit.Chem.Descriptors.NumHAcceptors.

**Wildman-Crippen LogP (MolLogP)**: This calculated LogP value can be used as a baseline estimate. RDKit: rdkit.Chem.Crippen.MolLogP.

V. Combinatorial and Contextual Rules

**Ratio of Polar to Non-polar Groups**: A high ratio of polar groups (amides, OH, etc.) to non-polar groups (alkyl chains, halogens) will generally lead to lower LogD.

**Balance of Aromaticity and Polarity**: A molecule with many aromatic rings but also many polar groups can have a moderate LogD. The effects can partially offset.

**"Lipophilic Efficiency"**: A molecule with few polar groups, and many non-polar groups, will have high lipophilic efficiency, and thus, high LogD.

**Nitrogen Heteroatom Count in Rings**: Molecules with 2 or more nitrogen atoms inside a 5 or 6-membered ring generally exhibit decreased LogD, as they show higher polarity and water solubility.
"""
}

rules_v2 = {
    "LogD": """
- Bulky alkyl groups (e.g., t-Bu, i-Pr) & long chains increase LogD.
- More aromatic rings, especially fused/rigid, increase LogD.
- Halogens (Cl, F, Br) increase LogD; multiple halogens add cumulatively.
- Ethers add polarity, slightly lowering LogD.
- Esters, amides, free carboxylic acids, protonated amines, & polar heterocycles lower LogD.
- Nitro, sulfonyl/sulfone, and other strong EW groups further lower LogD.
- Each extra polar/ionizable group shifts balance toward hydrophilicity.
- Intramolecular H-bonds can mask polarity, raising LogD.
- Steric shielding of polar sites (via bulky groups) increase LogD.
- Extended conjugation, rigid/planar, bridged/polycyclic structures, high sp² content, & symmetry increase LogD.
- Lower heteroatom:carbon and fewer nitrogen atoms favor higher LogD.
- Permanent charges (e.g., quaternary ammonium) drastically lower LogD.
- Ring size, saturation (aromatic > saturated), & substituent positions modulate LogD.
- If the SMILES contains “CC(C)(C)” (a tert‐butyl group) attached at a chiral center (e.g. “CC(C)(C)C@H”), then expect an increase in LogD by roughly 1–1.5 units compared to a similar scaffold lacking this group.
- If the molecule shows a fragment like “NC1=NC=NC2=C1C=C(C1=CN(…))N2” appended to a tert‐butyl bearing center, then LogD values typically fall in the ~3.1–3.2 range.
- If a structure features two or more fused aromatic rings (e.g. “C1=CC=C2…C2=C1”), then the overall lipophilicity is enhanced and LogD tends to be above 2.5.
- If an aromatic ring is directly substituted with halogens such as “Cl” or “F” (for instance, “C1=CC(Cl)=CC=C1”), then expect a boost in LogD by about 0.5 units relative to an unsubstituted ring.
- If multiple halogens are present on separate aromatic rings (e.g. both “Cl” and “F” on different rings), then the cumulative effect can raise LogD by around 1.0 unit or more.
- If the SMILES shows an “OCCOC” or “OCCOC2=” fragment (typical of ether linkers between aromatic rings), then these oxygenated linkers slightly reduce lipophilicity—often lowering LogD by 0.2–0.5 units compared to a direct aryl–aryl bond.
- If a sulfonyl fragment appears (i.e. “S(=O)(=O)”), especially appended to an aromatic or aliphatic segment, then LogD is reduced by roughly 0.5–1.0 units because of the high polarity of sulfonyl groups.
- If the structure contains an amide linkage (–C(=O)N–) not shielded by bulky groups, then expect a drop in LogD on the order of 0.3–0.7 units relative to analogous non‐amide linkers.
- If a free carboxylic acid group (“C(=O)O”) is present and not sterically hindered, then the compound tends to have a LogD that is 1–2 units lower than a similar ester or amide analogue.
- If a primary or secondary amine appears (e.g. “NC” or “N(C)C”) that is likely protonated at pH 7.4, then the molecule’s LogD will be lowered by approximately 1 unit relative to a neutral analogue.
- If a quaternary ammonium group (e.g. “N+(CH3)3”) is evident, then expect a dramatic drop in LogD – often resulting in near‐zero or negative values.
- If a heterocyclic aromatic ring contains a non‐protonated nitrogen (e.g. pyridine-like fragments such as “c1ccncc1”), then this feature tends to reduce LogD by 0.2–0.5 units relative to pure carbocyclic aromatics.
- If an extended conjugated system is present (for example, several aromatic rings linked by conjugated bonds), then the molecule often exhibits LogD values above 3 due to the cumulative hydrophobic surface.
- If polar groups (e.g. –OH, –NH2, –C(=O)O) are positioned so that intramolecular hydrogen bonding is likely, then their effective polarity is “masked” and LogD may be higher than predicted by counting polar groups alone.
- If a nitro group (–NO2) is present, then its strong electron‐withdrawing nature usually reduces LogD by about 1 unit or more.
- If an aromatic ether (Ar–O–Ar) is present rather than an aliphatic ether (R–O–R), then the effect on LogD is less pronounced—predict a LogD that is ~0.2–0.3 units higher than for an aliphatic ether analogue.
- If a carbonyl (C=O) is directly attached to an aromatic ring (as in an aromatic ketone), then expect a modest LogD reduction (around 0.3 units) versus a fully hydrocarbon substituted ring.
- If an alkyne is present in an aliphatic chain (e.g. “C#C” in “C#CCCC…”), then its low polarity means it contributes little to water solubility, keeping LogD relatively high.
- If the SMILES indicates a branched alkyl chain (such as an isopropyl group “CC(C)”) rather than a straight chain, then the branching typically increases LogD by enhancing hydrophobicity.
- If the structure contains multiple amide bonds in a row (e.g. a di- or tri-amide linker), then their combined polar effect can lower LogD by up to 1.5–2 units unless balanced by large lipophilic groups.
- If the molecule features a rigid bicyclic or polycyclic aromatic system (e.g. fused rings like “C1=CC2=CC=CC=C2C=C1”), then expect LogD values in the upper range (typically 2.5–4.0) because of the efficient stacking in lipophilic environments.
- If a spirocyclic motif is present, then the compact, three-dimensional arrangement often “hides” polar groups, leading to an increase in LogD by around 0.5 units relative to a more extended analogue.
- If the heterocycle is larger (e.g. a quinoline or isoquinoline instead of a pyridine), then the additional aromatic carbon atoms generally boost LogD by ~0.5–1.0 units.
- If bridging –CH2– groups are present between rings (e.g. “–CH2–” linking two aromatics), then these bridges increase hydrophobicity by reducing the effective polar surface area, raising LogD slightly.
- If the SMILES includes electron-withdrawing substituents (e.g. “CF3” or “Cl”) on an aromatic ring, then these groups reduce hydrogen-bonding capacity and typically increase LogD by about 0.3–0.7 units.
- If electron-donating groups (e.g. “OCH3”) are attached to an aromatic ring, then the increased polarity may lower LogD by ~0.2–0.5 units relative to a halogenated or unsubstituted ring.
- If a cyclic ether (e.g. a tetrahydrofuran ring represented as “C1CCOC1”) is incorporated, then its moderate polarity can reduce LogD by approximately 0.3–0.7 units compared to an all‐carbon ring of similar size.
- If the polar groups (such as amides or carboxyls) appear on the periphery of a large, lipophilic scaffold, then their impact on LogD is partially offset—resulting in LogD values that are about 0.5–1 unit higher than if the same polar groups were isolated.
- If steric hindrance is evident around a normally polar group (for example, an amide adjacent to a bulky tert‐butyl group), then the effective exposure of that polar functionality is reduced, leading to a higher LogD than predicted by polarity alone.
- If the overall structure can be roughly “fragmented” into lipophilic and hydrophilic pieces, then the net LogD is roughly additive. For example, two strongly lipophilic aromatic rings plus one polar amide might yield a predicted LogD in the range of 2.5–3.0, whereas replacing the amide with a carboxylic acid may drop the value by 1–1.5 units.
"""
} # 7029


rules_v3 = {
    "LogD": """
- Large, nonpolar substituents (e.g. tert‐butyl, isopropyl) increase lipophilicity.
- Extended hydrocarbon chains enhance nonpolar character and raise LogD.
- More aromatic rings typically provide significant hydrophobic surface area, increasing LogD.
- Rigid, fused rings tend to be more lipophilic than isolated aromatic rings.
- Incorporation of Cl, F, or Br generally increases lipophilicity due to their electron-withdrawing yet lipophilic nature.
- The cumulative effect of several halogen atoms can further raise LogD.
- Oxygen in ethers adds some polarity, often reducing LogD slightly compared to pure hydrocarbons.
- Esters introduce polarity through the carbonyl and adjacent oxygen, decreasing lipophilicity relative to nonpolar analogues.
- Amides are polar (and can hydrogen-bond), thus they tend to lower LogD.
- When deprotonated (or even in their neutral form), carboxyl groups strongly favor water, reducing LogD.
- Basic amine groups (if protonated at physiological pH) increase water solubility and lower LogD.
- Nitrogen-containing heterocycles that can be protonated tend to lower lipophilicity.
- Their strong electron-withdrawing nature and hydrogen-bonding capacity drive LogD down.
- Nitro substituents are strongly polar and tend to lower LogD.
- Each additional polar (or ionizable) group further shifts the balance toward hydrophilicity.
- When polar groups internally hydrogen bond, their “external” polarity is masked—potentially raising LogD.
- If bulky groups “shield” polar sites, the molecule may behave more lipophilically than expected.
- Extended conjugation (especially in fused aromatic systems) typically increases nonpolar character.
- Less-flexible, planar systems often pack well into lipophilic environments.
- Rigid, compact, polycyclic frameworks generally exhibit higher lipophilicity.
- Ketones and aldehydes add polarity (via the carbonyl), lowering LogD moderately compared to nonpolar groups.
- Molecules with a higher proportion of carbons versus polar heteroatoms tend to be more lipophilic.
- Fewer nitrogen atoms usually mean fewer opportunities for hydrogen bonding, favoring lipophilicity.
- Fixed positive or negative charges strongly favor aqueous solubility.
- Larger rings or macrocycles can be more lipophilic if they reduce the exposure of polar groups.
- Saturated (non-aromatic) rings tend to be less lipophilic than aromatic rings, all else being equal.
- Ortho, meta, or para placements affect steric hindrance and exposure of polar sites, thereby modulating lipophilicity.
- Delocalized electron density over a conjugated system can reduce the effective polarity, increasing LogD.
- More symmetric molecules may pack more efficiently in lipophilic environments, raising LogD.
- A higher proportion of sp²-hybridized carbons (e.g. in aromatic systems) usually boosts lipophilicity compared to sp³ centers that might bear polar substituents.
- If the SMILES contains “CC(C)(C)” (a tert‐butyl group) attached at a chiral center (e.g. “CC(C)(C)C@H”), then expect an increase in LogD by roughly 1–1.5 units compared to a similar scaffold lacking this group.
- If the molecule shows a fragment like “NC1=NC=NC2=C1C=C(C1=CN(…))N2” appended to a tert‐butyl bearing center, then LogD values typically fall in the ~3.1–3.2 range.
- If a structure features two or more fused aromatic rings (e.g. “C1=CC=C2…C2=C1”), then the overall lipophilicity is enhanced and LogD tends to be above 2.5.
- If an aromatic ring is directly substituted with halogens such as “Cl” or “F” (for instance, “C1=CC(Cl)=CC=C1”), then expect a boost in LogD by about 0.5 units relative to an unsubstituted ring.
- If multiple halogens are present on separate aromatic rings (e.g. both “Cl” and “F” on different rings), then the cumulative effect can raise LogD by around 1.0 unit or more.
- If the SMILES shows an “OCCOC” or “OCCOC2=” fragment (typical of ether linkers between aromatic rings), then these oxygenated linkers slightly reduce lipophilicity—often lowering LogD by 0.2–0.5 units compared to a direct aryl–aryl bond.
- If a sulfonyl fragment appears (i.e. “S(=O)(=O)”), especially appended to an aromatic or aliphatic segment, then LogD is reduced by roughly 0.5–1.0 units because of the high polarity of sulfonyl groups.
- If the structure contains an amide linkage (–C(=O)N–) not shielded by bulky groups, then expect a drop in LogD on the order of 0.3–0.7 units relative to analogous non‐amide linkers.
- If a free carboxylic acid group (“C(=O)O”) is present and not sterically hindered, then the compound tends to have a LogD that is 1–2 units lower than a similar ester or amide analogue.
- If a primary or secondary amine appears (e.g. “NC” or “N(C)C”) that is likely protonated at pH 7.4, then the molecule’s LogD will be lowered by approximately 1 unit relative to a neutral analogue.
- If a quaternary ammonium group (e.g. “N+(CH3)3”) is evident, then expect a dramatic drop in LogD – often resulting in near‐zero or negative values.
- If a heterocyclic aromatic ring contains a non‐protonated nitrogen (e.g. pyridine-like fragments such as “c1ccncc1”), then this feature tends to reduce LogD by 0.2–0.5 units relative to pure carbocyclic aromatics.
- If an extended conjugated system is present (for example, several aromatic rings linked by conjugated bonds), then the molecule often exhibits LogD values above 3 due to the cumulative hydrophobic surface.
- If polar groups (e.g. –OH, –NH2, –C(=O)O) are positioned so that intramolecular hydrogen bonding is likely, then their effective polarity is “masked” and LogD may be higher than predicted by counting polar groups alone.
- If a nitro group (–NO2) is present, then its strong electron‐withdrawing nature usually reduces LogD by about 1 unit or more.
- If an aromatic ether (Ar–O–Ar) is present rather than an aliphatic ether (R–O–R), then the effect on LogD is less pronounced—predict a LogD that is ~0.2–0.3 units higher than for an aliphatic ether analogue.
- If a carbonyl (C=O) is directly attached to an aromatic ring (as in an aromatic ketone), then expect a modest LogD reduction (around 0.3 units) versus a fully hydrocarbon substituted ring.
- If an alkyne is present in an aliphatic chain (e.g. “C#C” in “C#CCCC…”), then its low polarity means it contributes little to water solubility, keeping LogD relatively high.
- If the SMILES indicates a branched alkyl chain (such as an isopropyl group “CC(C)”) rather than a straight chain, then the branching typically increases LogD by enhancing hydrophobicity.
- If the structure contains multiple amide bonds in a row (e.g. a di- or tri-amide linker), then their combined polar effect can lower LogD by up to 1.5–2 units unless balanced by large lipophilic groups.
- If the molecule features a rigid bicyclic or polycyclic aromatic system (e.g. fused rings like “C1=CC2=CC=CC=C2C=C1”), then expect LogD values in the upper range (typically 2.5–4.0) because of the efficient stacking in lipophilic environments.
- If a spirocyclic motif is present, then the compact, three-dimensional arrangement often “hides” polar groups, leading to an increase in LogD by around 0.5 units relative to a more extended analogue.
- If the heterocycle is larger (e.g. a quinoline or isoquinoline instead of a pyridine), then the additional aromatic carbon atoms generally boost LogD by ~0.5–1.0 units.
- If bridging –CH2– groups are present between rings (e.g. “–CH2–” linking two aromatics), then these bridges increase hydrophobicity by reducing the effective polar surface area, raising LogD slightly.
- If the SMILES includes electron-withdrawing substituents (e.g. “CF3” or “Cl”) on an aromatic ring, then these groups reduce hydrogen-bonding capacity and typically increase LogD by about 0.3–0.7 units.
- If electron-donating groups (e.g. “OCH3”) are attached to an aromatic ring, then the increased polarity may lower LogD by ~0.2–0.5 units relative to a halogenated or unsubstituted ring.
- If a cyclic ether (e.g. a tetrahydrofuran ring represented as “C1CCOC1”) is incorporated, then its moderate polarity can reduce LogD by approximately 0.3–0.7 units compared to an all‐carbon ring of similar size.
- If the polar groups (such as amides or carboxyls) appear on the periphery of a large, lipophilic scaffold, then their impact on LogD is partially offset—resulting in LogD values that are about 0.5–1 unit higher than if the same polar groups were isolated.
- If steric hindrance is evident around a normally polar group (for example, an amide adjacent to a bulky tert‐butyl group), then the effective exposure of that polar functionality is reduced, leading to a higher LogD than predicted by polarity alone.
- If the overall structure can be roughly “fragmented” into lipophilic and hydrophilic pieces, then the net LogD is roughly additive. For example, two strongly lipophilic aromatic rings plus one polar amide might yield a predicted LogD in the range of 2.5–3.0, whereas replacing the amide with a carboxylic acid may drop the value by 1–1.5 units.
"""
}


rules_v4 = {
    "LogD": """
- If the SMILES contains “CC(C)(C)” (a tert‐butyl group) attached at a chiral center (e.g. “CC(C)(C)C@H”), then expect an increase in LogD by roughly 1–1.5 units compared to a similar scaffold lacking this group.
- If the molecule shows a fragment like “NC1=NC=NC2=C1C=C(C1=CN(…))N2” appended to a tert‐butyl bearing center, then LogD values typically fall in the ~3.1–3.2 range.
- If a structure features two or more fused aromatic rings (e.g. “C1=CC=C2…C2=C1”), then the overall lipophilicity is enhanced and LogD tends to be above 2.5.
- If an aromatic ring is directly substituted with halogens such as “Cl” or “F” (for instance, “C1=CC(Cl)=CC=C1”), then expect a boost in LogD by about 0.5 units relative to an unsubstituted ring.
- If multiple halogens are present on separate aromatic rings (e.g. both “Cl” and “F” on different rings), then the cumulative effect can raise LogD by around 1.0 unit or more.
- If the SMILES shows an “OCCOC” or “OCCOC2=” fragment (typical of ether linkers between aromatic rings), then these oxygenated linkers slightly reduce lipophilicity—often lowering LogD by 0.2–0.5 units compared to a direct aryl–aryl bond.
- If a sulfonyl fragment appears (i.e. “S(=O)(=O)”), especially appended to an aromatic or aliphatic segment, then LogD is reduced by roughly 0.5–1.0 units because of the high polarity of sulfonyl groups.
- If the structure contains an amide linkage (–C(=O)N–) not shielded by bulky groups, then expect a drop in LogD on the order of 0.3–0.7 units relative to analogous non‐amide linkers.
- If a free carboxylic acid group (“C(=O)O”) is present and not sterically hindered, then the compound tends to have a LogD that is 1–2 units lower than a similar ester or amide analogue.
- If a primary or secondary amine appears (e.g. “NC” or “N(C)C”) that is likely protonated at pH 7.4, then the molecule’s LogD will be lowered by approximately 1 unit relative to a neutral analogue.
- If a quaternary ammonium group (e.g. “N+(CH3)3”) is evident, then expect a dramatic drop in LogD – often resulting in near‐zero or negative values.
- If a heterocyclic aromatic ring contains a non‐protonated nitrogen (e.g. pyridine-like fragments such as “c1ccncc1”), then this feature tends to reduce LogD by 0.2–0.5 units relative to pure carbocyclic aromatics.
- If an extended conjugated system is present (for example, several aromatic rings linked by conjugated bonds), then the molecule often exhibits LogD values above 3 due to the cumulative hydrophobic surface.
- If polar groups (e.g. –OH, –NH2, –C(=O)O) are positioned so that intramolecular hydrogen bonding is likely, then their effective polarity is “masked” and LogD may be higher than predicted by counting polar groups alone.
- If a nitro group (–NO2) is present, then its strong electron‐withdrawing nature usually reduces LogD by about 1 unit or more.
- If an aromatic ether (Ar–O–Ar) is present rather than an aliphatic ether (R–O–R), then the effect on LogD is less pronounced—predict a LogD that is ~0.2–0.3 units higher than for an aliphatic ether analogue.
- If a carbonyl (C=O) is directly attached to an aromatic ring (as in an aromatic ketone), then expect a modest LogD reduction (around 0.3 units) versus a fully hydrocarbon substituted ring.
- If an alkyne is present in an aliphatic chain (e.g. “C#C” in “C#CCCC…”), then its low polarity means it contributes little to water solubility, keeping LogD relatively high.
- If the SMILES indicates a branched alkyl chain (such as an isopropyl group “CC(C)”) rather than a straight chain, then the branching typically increases LogD by enhancing hydrophobicity.
- If the structure contains multiple amide bonds in a row (e.g. a di- or tri-amide linker), then their combined polar effect can lower LogD by up to 1.5–2 units unless balanced by large lipophilic groups.
- If the molecule features a rigid bicyclic or polycyclic aromatic system (e.g. fused rings like “C1=CC2=CC=CC=C2C=C1”), then expect LogD values in the upper range (typically 2.5–4.0) because of the efficient stacking in lipophilic environments.
- If a spirocyclic motif is present, then the compact, three-dimensional arrangement often “hides” polar groups, leading to an increase in LogD by around 0.5 units relative to a more extended analogue.
- If the heterocycle is larger (e.g. a quinoline or isoquinoline instead of a pyridine), then the additional aromatic carbon atoms generally boost LogD by ~0.5–1.0 units.
- If bridging –CH2– groups are present between rings (e.g. “–CH2–” linking two aromatics), then these bridges increase hydrophobicity by reducing the effective polar surface area, raising LogD slightly.
- If the SMILES includes electron-withdrawing substituents (e.g. “CF3” or “Cl”) on an aromatic ring, then these groups reduce hydrogen-bonding capacity and typically increase LogD by about 0.3–0.7 units.
- If electron-donating groups (e.g. “OCH3”) are attached to an aromatic ring, then the increased polarity may lower LogD by ~0.2–0.5 units relative to a halogenated or unsubstituted ring.
- If a cyclic ether (e.g. a tetrahydrofuran ring represented as “C1CCOC1”) is incorporated, then its moderate polarity can reduce LogD by approximately 0.3–0.7 units compared to an all‐carbon ring of similar size.
- If the polar groups (such as amides or carboxyls) appear on the periphery of a large, lipophilic scaffold, then their impact on LogD is partially offset—resulting in LogD values that are about 0.5–1 unit higher than if the same polar groups were isolated.
- If steric hindrance is evident around a normally polar group (for example, an amide adjacent to a bulky tert‐butyl group), then the effective exposure of that polar functionality is reduced, leading to a higher LogD than predicted by polarity alone.
- If the overall structure can be roughly “fragmented” into lipophilic and hydrophilic pieces, then the net LogD is roughly additive. For example, two strongly lipophilic aromatic rings plus one polar amide might yield a predicted LogD in the range of 2.5–3.0, whereas replacing the amide with a carboxylic acid may drop the value by 1–1.5 units.
"""
}

rules = {
    "LogD": """
- Bulky alkyl groups (e.g., t-Bu, i-Pr) & long chains ↑ LogD.
- More aromatic rings, especially fused/rigid, ↑ LogD.
- Halogens (Cl, F, Br) increase LogD; multiple halogens add cumulatively.
- Ethers add polarity, slightly lowering LogD.
- Esters, amides, free carboxylic acids, protonated amines, & polar heterocycles ↓ LogD.
- Nitro, sulfonyl/sulfone, and other strong EW groups further ↓ LogD.
- Each extra polar/ionizable group shifts balance toward hydrophilicity.
- Intramolecular H-bonds can mask polarity, raising LogD.
- Steric shielding of polar sites (via bulky groups) ↑ LogD.
- Extended conjugation, rigid/planar, bridged/polycyclic structures, high sp² content, & symmetry ↑ LogD.
- Lower heteroatom:carbon and fewer nitrogen atoms favor higher LogD.
- Permanent charges (e.g., quaternary ammonium) drastically ↓ LogD.
- Ring size, saturation (aromatic > saturated), & substituent positions modulate LogD.
- Specific examples:
- t-Bu at a chiral center: +1–1.5 units.
- t-Bu fragments: LogD ~3.1–3.2.
- Fused aromatics: LogD >2.5.
- Aromatic halogenation: ~+0.5; on separate rings: ~+1+ cumulatively.
- Ether linkers (e.g., OCCOC): –0.2–0.5.
- Sulfonyl fragments: –0.5–1.0.
- Amide linkages: –0.3–0.7 (multiple amides: –1.5–2).
- Free acids: –1–2.
- Primary/secondary amines (protonated): ~–1.
- Quaternary ammonium: near 0 or negative.
- Non-protonated aromatic N: –0.2–0.5.
- Extended conjugation: LogD >3.
- Nitro groups: –~1+.
- Aromatic ethers vs aliphatic: +0.2–0.3 difference.
- Aromatic carbonyls: –~0.3.
- Alkynes: minimal effect.
- Branched chains (e.g., isopropyl) ↑ LogD.
- Rigid bicyclic/polycyclic aromatics: LogD ~2.5–4.0.
- Spirocycles: +~0.5.
- Larger heterocycles (quinoline/isoquinoline): +0.5–1.0.
- CH2 bridges between rings: slight ↑.
- EWGs (CF3, Cl) on aromatics: +0.3–0.7.
- EDGs (OCH3): –0.2–0.5.
- Cyclic ethers (e.g., THF): –0.3–0.7.
- Polar groups on a large lipophilic scaffold: impact partly offset (+0.5–1).
- Steric hindrance around polar groups: ↑ LogD.
- Overall, net LogD is roughly additive: sum(lipophilic fragments) minus sum(polar effects).
"""
}

rules_v5 = {
"""
t-Bu at a chiral center: +1–1.5 units; t-Bu fragments: LogD ~3.1–3.2; Fused aromatics: LogD >2.5; Aromatic halogenation: ~+0.5; on separate rings: ~+1+ cumulatively; Ether linkers (e.g., OCCOC): –0.2–0.5; Sulfonyl fragments: –0.5–1.0; Amide linkages: –0.3–0.7 (multiple amides: –1.5–2); Free acids: –1–2.
"""
}

# Primary/secondary amines (protonated): ~–1; Quaternary ammonium: near 0 or negative; Non-protonated aromatic N: –0.2–0.5; Extended conjugation: LogD >3; Nitro groups: –~1+; Aromatic ethers vs aliphatic: +0.2–0.3 difference; Aromatic carbonyls: –~0.3; Alkynes: minimal effect; Branched chains (e.g., isopropyl) ↑ LogD; Rigid bicyclic/polycyclic aromatics: LogD ~2.5–4.0; Spirocycles: +~0.5; Larger heterocycles (quinoline/isoquinoline): +0.5–1.0.


