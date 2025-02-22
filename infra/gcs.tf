resource "google_storage_bucket" "training-set" {
  name     = "${data.google_project.creature.project_id}-training-set"
  location = var.region
  project  = data.google_project.creature.project_id

  force_destroy               = false
  uniform_bucket_level_access = true

}
