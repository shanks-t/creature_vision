# resource "google_container_cluster" "primary" {
#   name     = "ml-cv-cluster"
#   location = var.zone

#   # Enable the free control plane by using DNS-based endpoint
#   release_channel {
#     channel = "REGULAR"
#   }

#   # Remove default node pool
#   remove_default_node_pool = true
#   initial_node_count       = 1

#   # Disable cluster autoscaling since we're optimizing for minimum cost
#   # You can enable it later when needed
#   cluster_autoscaling {
#     enabled = false
#   }

#   # Keep Workload Identity for Vertex AI integration
#   workload_identity_config {
#     workload_pool = "${var.project_id}.svc.id.goog"
#   }

#   # Network configuration
#   network    = google_compute_network.vpc.name
#   subnetwork = google_compute_subnetwork.subnet.name

#   depends_on = [
#     google_compute_network.vpc,
#     google_compute_subnetwork.subnet
#   ]
# }

# resource "google_container_node_pool" "primary_nodes" {
#   name     = "ml-cv-pool"
#   location = var.zone
#   cluster  = google_container_cluster.primary.name

#   # Single node configuration with minimum autoscaling
#   autoscaling {
#     min_node_count = 0
#     max_node_count = 1
#   }

#   node_config {
#     # Use n2-standard-2 for better pricing
#     machine_type = "n2-standard-2"

#     # Enable Spot VMs for cost savings
#     spot = true

#     # Preemptible is an older version of Spot VMs, use spot instead
#     # preemptible = true

#     oauth_scopes = [
#       "https://www.googleapis.com/auth/cloud-platform"
#     ]

#     # Keep Workload Identity
#     workload_metadata_config {
#       mode = "GKE_METADATA"
#     }

#     # Optional: Add labels to track spot instances
#     labels = {
#       "cloud.google.com/gke-spot" = "true"
#     }
#   }

#   # Recommended for Spot VMs to handle preemption gracefully
#   management {
#     auto_repair  = true
#     auto_upgrade = true
#   }

#   depends_on = [
#     google_container_cluster.primary
#   ]
# }
