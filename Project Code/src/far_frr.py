import numpy as np
import matplotlib.pyplot as plt


similarity_scores = np.random.rand(100)
true_labels = np.random.choice([0, 1], size=100)  # 1 for true positive, 0 for impostors

# Function to calculate FAR and FRR for a range of thresholds
def compute_far_frr(similarity_scores, true_labels, thresholds):
    far = []
    frr = []

    for threshold in thresholds:
        # False Acceptances (FAR): Predict impostors as legitimate (threshold < score)
        false_acceptances = np.sum((similarity_scores >= threshold) & (true_labels == 0))
        far.append(false_acceptances / np.sum(true_labels == 0))  # False acceptance rate

        # False Rejections (FRR): Predict genuine users as impostors (threshold > score)
        false_rejections = np.sum((similarity_scores < threshold) & (true_labels == 1))
        frr.append(false_rejections / np.sum(true_labels == 1))  # False rejection rate

    return np.array(far), np.array(frr)

# Define thresholds to evaluate
thresholds = np.linspace(0, 1, 100)

# Calculate FAR and FRR across thresholds
far, frr = compute_far_frr(similarity_scores, true_labels, thresholds)

# Plot FAR vs FRR
plt.figure(figsize=(8, 6))
plt.plot(far, frr, marker='o', linestyle='-', color='b')
plt.axvline(x=0.75, color='r', linestyle='--')  # Highlight the threshold at 0.75
plt.title('FAR vs FRR Trade-off Curve')
plt.xlabel('False Acceptance Rate (FAR)')
plt.ylabel('False Rejection Rate (FRR)')
plt.grid(True)
plt.legend(['FAR-FRR Curve', 'Threshold (0.75)'])
plt.show()
