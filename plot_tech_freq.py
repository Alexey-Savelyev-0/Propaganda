import matplotlib.pyplot as plt

# Mapping of technique names to their corresponding indices
technique_to_index = {
    'Appeal_to_Authority': 4,
    'Appeal_to_fear-prejudice': 6,
    'Bandwagon,Reductio_ad_hitlerum': 8,
    'Black-and-White_Fallacy': 11,
    'Causal_Oversimplification': 2,
    'Doubt': 13,
    'Exaggeration,Minimisation': 7,
    'Flag-Waving': 0,
    'Loaded_Language': 3,
    'Name_Calling,Labeling': 1,
    'Repetition': 10,
    'Slogans': 5,
    'Thought-terminating_Cliches': 9,
    'Whataboutism,Straw_Men,Red_Herring': 12
}

# List of counts corresponding to each technique index in the training data
counts = [152, 530, 122, 1246, 100, 78, 231, 271, 52, 61, 474, 93, 88, 265]

techniques = [None] * len(counts)
for technique, index in technique_to_index.items():
    techniques[index] = technique

# Plotting the bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(techniques, counts, color='skyblue')
plt.xlabel('Propaganda Technique')
plt.ylabel('Frequency')
plt.title('Frequency of Propaganda Techniques')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('technique_frequencies.png', dpi=300, bbox_inches='tight')
plt.close()
