# Create a new GCP project
data "google_project" "creature" {
}


# Enable required APIs
resource "google_project_service" "enabled_apis" {
  for_each = toset([
    "run.googleapis.com",
    "bigquery.googleapis.com",
    "cloudfunctions.googleapis.com",
    "storage.googleapis.com",
    "aiplatform.googleapis.com",
    "containerregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "datacatalog.googleapis.com",
    "datastore.googleapis.com",
    "compute.googleapis.com",
    "container.googleapis.com",
  ])

  project = data.google_project.creature.project_id
  service = each.key

  disable_on_destroy = true
}

# Create a service account for Terraform
resource "google_service_account" "terraform_sa" {
  account_id   = "terraform-sa"
  display_name = "Terraform Service Account"
  project      = data.google_project.creature.project_id
}

# Grant the service account necessary permissions
resource "google_project_iam_member" "terraform_sa_roles" {
  for_each = toset([
    "roles/editor",
    "roles/run.admin",
    "roles/storage.admin",
    "roles/bigquery.admin",
    "roles/cloudfunctions.admin",
    "roles/aiplatform.admin",
    "roles/artifactregistry.admin"
  ])

  project = data.google_project.creature.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.terraform_sa.email}"
}
