# gke.tf
resource "google_container_cluster" "primary" {
  name     = "ml-cv-cluster"
  location = var.zone

  # Use a zonal cluster to reduce costs
  remove_default_node_pool = true
  initial_node_count       = 1

  # Enable Workload Identity for Vertex AI integration
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Network configuration
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name
}

resource "google_container_node_pool" "primary_nodes" {
  name     = "ml-cv-pool"
  location = var.zone
  cluster  = google_container_cluster.primary.name

  # Autoscaling configuration
  autoscaling {
    min_node_count = 0
    max_node_count = 3
  }

  node_config {
    machine_type = "n1-standard-4"

    # Enable GPU if needed
    # guest_accelerator {
    #   type  = "nvidia-tesla-t4"
    #   count = 1
    # }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    # Enable Workload Identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
  }
}
