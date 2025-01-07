import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

# Define file paths
data_file_path = '/storage/emulated/0/data_EEG/rest/combined_data_rest.edf'

pdf_filename = '/storage/emulated/0/data_EEG/rest/results_combined_rest_final2.pdf'
jpg_dir = '/storage/emulated/0/data_EEG/rest/jpg_images/'

# Create directory for JPGs if it doesn't exist
if not os.path.exists(jpg_dir):
    os.makedirs(jpg_dir)

# Load the EEG data from the .edf file using MNE
raw_data = mne.io.read_raw_edf(data_file_path, preload=True)
raw_data.set_eeg_reference('average', projection=True)

# Preprocess the EEG data
theta_band = raw_data.copy().filter(4, 8, method='iir')
alpha_band = raw_data.copy().filter(8, 13, method='iir')
beta_band = raw_data.copy().filter(13, 30, method='iir')

eeg_channels = raw_data.pick_types(eeg=True)
eeg_data, _ = eeg_channels[:, :]

# Feature Extraction - Power Spectral Density (PSD) using Welch's method
def compute_psd(signal, fs=256, nperseg=256):
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    return psd

theta_psd = np.array([compute_psd(signal) for signal in theta_band.get_data()])
alpha_psd = np.array([compute_psd(signal) for signal in alpha_band.get_data()])
beta_psd = np.array([compute_psd(signal) for signal in beta_band.get_data()])
psd_data = np.hstack((theta_psd, alpha_psd, beta_psd))

# Split the data into 70% (A) and 30% (B)
psd_data_A, psd_data_B = train_test_split(psd_data, test_size=0.3, random_state=42)

# Function to process EEG data
def process_eeg_data(psd_data, dataset_name, pdf, jpg_dir):
    # Outlier Detection using Z-Score
    z_scores = np.abs(stats.zscore(psd_data))
    threshold = 3
    psd_data_clean = psd_data[(z_scores < threshold).all(axis=1)]

    # Data Augmentation (Noise Injection)
    def augment_data(signal, noise_factor=0.01):
        noise = noise_factor * np.random.normal(size=signal.shape)
        augmented_signal = signal + noise
        return augmented_signal

    augmented_data = np.array([augment_data(signal) for signal in psd_data_clean])
    psd_data_combined = np.vstack((psd_data_clean, augmented_data))

    # K-Means Clustering (3 clusters for low, moderate, and high stress)
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(psd_data_combined)

    # Refining Feature Extraction: Extract statistical features (mean, std, etc.)
    def extract_features(psd):
        return [np.mean(psd), np.std(psd), np.max(psd), np.min(psd), np.median(psd)]

    features = np.array([extract_features(signal) for signal in psd_data_combined])

    # Train-Test Split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, clusters, test_size=0.3, random_state=42)

    # SVM Classification
    svm_model = SVC(kernel='poly', degree=3, random_state=42)
    svm_model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # Cross-Validation
    cv_scores = cross_val_score(svm_model, X_scaled, clusters, cv=10)

    # Create text summary
    results_summary = f"""
    Model Evaluation ({dataset_name}):
    -------------------------
    Accuracy: {accuracy * 100:.2f}%
    F1-Score: {f1:.2f}
    Recall: {recall:.2f}

    Confusion Matrix:
    {conf_matrix}

    10-fold Cross-Validation Accuracy: {np.mean(cv_scores) * 100:.2f}%
    """

    # Plot Confusion Matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(conf_matrix, cmap='Blues')
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.colorbar()
    plt.xlabel('Predicted', fontweight='bold', fontsize=20)
    plt.ylabel('Actual', fontweight='bold', fontsize=20)
    plt.xticks(fontweight='heavy', fontsize=20)
    plt.yticks(fontweight='heavy', fontsize=20)
    plt.tight_layout()
    pdf.savefig()  # save to PDF
    plt.savefig(f'{jpg_dir}/confusion_matrix_{dataset_name}.jpg', dpi=300)
    plt.close()

    # Plot K-Means Clusters using PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', marker='o', edgecolor='k')
    plt.title(f"K-Means Clustering of EEG Data - {dataset_name}")
    plt.xlabel('PCA Component 1', fontweight='bold', fontsize=20)
    plt.ylabel('PCA Component 2', fontweight='bold', fontsize=20)
    plt.xticks(fontweight='heavy', fontsize=20)
    plt.yticks(fontweight='heavy', fontsize=20)
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    pdf.savefig()  # save to PDF
    plt.savefig(f'{jpg_dir}/kmeans_pca_{dataset_name}.jpg', dpi=300)
    plt.close()

    # Plot Accuracy over Cross-Validation Folds
    plt.figure(figsize=(12, 10))
    plt.plot(range(1, 11), cv_scores * 100, marker='o')
    plt.title(f'SVM Cross-Validation Accuracy (10-Folds) - {dataset_name}')
    plt.xlabel('Fold', fontweight='bold', fontsize=20)
    plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=20)
    plt.xticks(fontweight='heavy', fontsize=20)
    plt.yticks(fontweight='heavy', fontsize=20)
    plt.ylim([80, 100])
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()  # save to PDF
    plt.savefig(f'{jpg_dir}/cross_validation_{dataset_name}.jpg', dpi=300)
    plt.close()

    # Save results summary as a PDF page
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.1, 0.9, results_summary, fontsize=12, verticalalignment='top')
    ax.axis('off')
    pdf.savefig(fig)
    plt.close()

    return accuracy, f1, recall

# Create a single PdfPages instance to save all figures
with PdfPages(pdf_filename) as pdf:
    # Process Data A
    acc_A, f1_A, recall_A = process_eeg_data(psd_data_A, "Data A", pdf, jpg_dir)
    # Process Data B
    acc_B, f1_B, recall_B = process_eeg_data(psd_data_B, "Data B", pdf, jpg_dir)

    # Calculate Overall Results
    overall_acc = (acc_A + acc_B) / 2
    overall_f1 = (f1_A + f1_B) / 2
    overall_recall = (recall_A + recall_B) / 2

    # Create summary for Overall results
    overall_summary = f"""
    Overall Model Evaluation:
    -------------------------
    Accuracy: {overall_acc * 100:.2f}%
    F1-Score: {overall_f1:.2f}
    Recall: {overall_recall:.2f}
    """

    # Save overall results summary as a PDF page
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.1, 0.9, overall_summary, fontsize=14, verticalalignment='top', fontweight='bold')
    ax.axis('off')
    pdf.savefig(fig)
    plt.close()

    # Overall Confusion Matrix for the whole data set (Data A + Data B)
    combined_psd_data = np.vstack((psd_data_A, psd_data_B))
    combined_clusters = KMeans(n_clusters=3, random_state=0).fit_predict(combined_psd_data)
    
    # Refining Feature Extraction: Extract statistical features (mean, std, etc.)
    def extract_features(psd):
        return [np.mean(psd), np.std(psd), np.max(psd), np.min(psd), np.median(psd)]

    combined_features = np.array([extract_features(signal) for signal in combined_psd_data])
    X_combined_scaled = StandardScaler().fit_transform(combined_features)
    
    X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
        X_combined_scaled, combined_clusters, test_size=0.3, random_state=42)

    svm_model_combined = SVC(kernel='poly', degree=3, random_state=42)
    svm_model_combined.fit(X_train_combined, y_train_combined)
    
    y_pred_combined = svm_model_combined.predict(X_test_combined)
    combined_conf_matrix = confusion_matrix(y_test_combined, y_pred_combined)

    # Plot Overall Confusion Matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(combined_conf_matrix, cmap='Blues')
    plt.title(f"Overall Confusion Matrix (Data A + Data B)", fontweight='bold', fontsize=20)
    plt.colorbar()
    plt.xlabel('Predicted', fontweight='bold', fontsize=20)
    plt.ylabel('Actual', fontweight='bold', fontsize=20)
    plt.xticks(fontweight='bold', fontsize=20)
    plt.yticks(fontweight='bold', fontsize=20)
    plt.tight_layout()
    pdf.savefig()  # save to PDF
    plt.savefig(f'{jpg_dir}/confusion_matrix_combined.jpg', dpi=300)
    plt.close()

    # PCA and KMeans Clustering for the Combined Dataset
    pca_combined = PCA(n_components=2)
    pca_result_combined = pca_combined.fit_transform(X_combined_scaled)

    plt.figure(figsize=(12, 10))
    plt.scatter(pca_result_combined[:, 0], pca_result_combined[:, 1], c=combined_clusters, cmap='viridis', marker='o', edgecolor='k')
    plt.title(f"K-Means Clustering of Combined EEG Data (Data A + Data B)", fontweight='bold', fontsize=20)
    plt.xlabel('PCA Component 1', fontweight='bold', fontsize=20)
    plt.ylabel('PCA Component 2', fontweight='bold', fontsize=20)
    plt.xticks(fontweight='bold', fontsize=20)
    plt.yticks(fontweight='bold', fontsize=20)
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    pdf.savefig()  # save to PDF
    plt.savefig(f'{jpg_dir}/kmeans_pca_combined.jpg', dpi=300)
    plt.close()

print("Figures and results saved successfully.") h
