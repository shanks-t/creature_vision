# vpc.tf
resource "google_compute_network" "vpc" {
  name                    = "ml-cv-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet" {
  name          = "ml-cv-subnet"
  region        = var.region
  network       = google_compute_network.vpc.name
  ip_cidr_range = "10.0.0.0/16"
}
