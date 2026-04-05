from __future__ import annotations

import torch



def _collect_embeddings_and_labels(model, validation_loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in validation_loader:
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
            embeddings = model.encode(texts, convert_to_tensor=True, device=device)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

    if not all_embeddings:
        raise ValueError("Validation loader is empty, cannot evaluate model.")

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(embeddings, embeddings.t())
    return similarity_matrix, labels



def evaluate_top_k(model, validation_loader, device, k=1):
    similarity_matrix, all_labels = _collect_embeddings_and_labels(model, validation_loader, device)

    correct_matches = 0
    total_matches = 0

    for i in range(len(all_labels)):
        similarity = similarity_matrix[i].clone()
        similarity[i] = -1
        top_k_indices = torch.topk(similarity, k).indices
        correct_matches += int(all_labels[i] in all_labels[top_k_indices])
        total_matches += 1

    return correct_matches / total_matches if total_matches > 0 else 0.0



def evaluate_recall_at_ground_truth(model, validation_loader, device):
    similarity_matrix, all_labels = _collect_embeddings_and_labels(model, validation_loader, device)

    correct_matches = 0
    total_matches = 0

    for i in range(len(all_labels)):
        true_label = all_labels[i]
        k = int(torch.sum(all_labels == true_label).item()) - 1
        if k <= 0:
            continue

        similarity = similarity_matrix[i].clone()
        similarity[i] = -1
        top_k_indices = torch.topk(similarity, k).indices

        correct_matches += int(torch.sum(all_labels[top_k_indices] == true_label).item())
        total_matches += k

    return correct_matches / total_matches if total_matches > 0 else 0.0



def evaluate_metrics(model, validation_loader, device, fixed_k=1):
    """
    Returns:
        accuracy_at_fixed_k,
        recall_at_ground_truth,
        mean_reciprocal_rank
    """
    similarity_matrix, all_labels = _collect_embeddings_and_labels(model, validation_loader, device)

    correct_matches_fixed_k = 0
    total_matches_fixed_k = 0
    correct_matches_recall = 0
    total_matches_recall = 0
    reciprocal_rank_sum = 0.0

    for i in range(len(all_labels)):
        similarity = similarity_matrix[i].clone()
        similarity[i] = -1
        true_label = all_labels[i]

        top_k_indices_fixed = torch.topk(similarity, fixed_k).indices
        correct_matches_fixed_k += int(true_label in all_labels[top_k_indices_fixed])
        total_matches_fixed_k += 1

        ground_truth_k = int(torch.sum(all_labels == true_label).item()) - 1
        if ground_truth_k > 0:
            top_k_indices_recall = torch.topk(similarity, ground_truth_k).indices
            correct_matches_recall += int(torch.sum(all_labels[top_k_indices_recall] == true_label).item())
            total_matches_recall += ground_truth_k

        ranked_indices = torch.argsort(similarity, descending=True)
        positive_mask = all_labels[ranked_indices] == true_label
        positive_positions = torch.nonzero(positive_mask, as_tuple=False)
        if len(positive_positions) > 0:
            best_rank = int(positive_positions[0].item()) + 1
            reciprocal_rank_sum += 1.0 / best_rank

    accuracy = correct_matches_fixed_k / total_matches_fixed_k if total_matches_fixed_k > 0 else 0.0
    recall_at_ground_truth = correct_matches_recall / total_matches_recall if total_matches_recall > 0 else 0.0
    mean_reciprocal_rank = reciprocal_rank_sum / len(all_labels) if len(all_labels) > 0 else 0.0

    return accuracy, recall_at_ground_truth, mean_reciprocal_rank
