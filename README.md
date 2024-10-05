1. Local Development and Testing:[✅]
   - Continue developing and testing your Python app locally.
   - Run Prometheus in a container to collect metrics from your app.
   - Add Grafana as a container to visualize metrics.
   - Use Docker Compose to manage these containers locally.

2. Containerize Everything:[✅]
   - Ensure your Python app, Prometheus, and Grafana are all containerized.
   - Test the entire stack using Docker Compose on your MacBook.

3. Kubernetes Basics:
   - Install minikube or kind on your MacBook to create a local Kubernetes cluster.
   - Learn Kubernetes basics: pods, services, deployments, configmaps, etc.

4. Migrate to Local Kubernetes:
   - Convert your Docker Compose setup to Kubernetes manifests.
   - Deploy your app, Prometheus, and Grafana to your local Kubernetes cluster.
   - Test thoroughly in this environment.

5. Set Up Homelab:
   - Install Proxmox on your homelab server if not already done.
   - Create VMs for your Kubernetes nodes (master and workers).

6. Install Kubernetes in Homelab:
   - Set up a Kubernetes cluster on your Proxmox VMs.
   - Options include kubeadm, k3s, or a distribution like Rancher.

7. Deploy to Homelab Kubernetes:
   - Push your container images to a container registry (e.g., Docker Hub).
   - Apply your Kubernetes manifests to your homelab cluster.
   - Test and verify everything is working as expected.

8. Implement CI/CD:
   - Set up a CI/CD pipeline to automate deployments to your homelab cluster.

9. Monitoring and Management:
   - Implement cluster-wide monitoring and logging solutions.
   - Set up backup and disaster recovery processes.

### Configure Internal Network
- use dedicated network for K8s nodes

1. Start a new shell session on your Proxmox VE server and switch to the Root shell if you haven’t already
    ```
    sudo -i
    ```
2. Take a backup of the network configuration file /etc/network/interfaces.
    ```
   cp /etc/network/interfaces /etc/network/interfaces.original
    ```
3. Open the /etc/network/interfaces file in a text editor and append the below configuration for the new network vmbr1.
    ```
   # /etc/network/interfaces
    ...
    ...
    # Dedicated internal network for Kubernetes cluster
    auto vmbr1
    iface vmbr1 inet static
        address  10.0.1.1/24
        bridge-ports none
        bridge-stp off
        bridge-fd 0

        post-up   echo 1 > /proc/sys/net/ipv4/ip_forward
        post-up   iptables -t nat -A POSTROUTING -s '10.0.1.0/24' -o vmbr0 -j MASQUERADE
        post-down iptables -t nat -D POSTROUTING -s '10.0.1.0/24' -o vmbr0 -j MASQUERADE
    ```

### Prepare VM Template
- https://austinsnerdythings.com/2021/08/30/how-to-create-a-proxmox-ubuntu-cloud-init-image/

1. Run script to setup template:
    ```
    bash -c "$(wget qLO - https://raw.githubusercontent.com/shanks-t/creature_vision/refs/heads/main/scripts/create-vm-templ.sh)"
    ```

### Generate SSH key pair

1. Generate an SSH key pair and save it to the specified directory.
    ```
    ssh-keygen -t rsa -b 4096 -f ~/proxmox-kubernetes/ssh-keys/id_rsa -C "k8s-admin@cluster.local"
    ```

### Setup bastion host

1. Create a new VM by cloning the VM template we’ve just created.
    ```
    qm clone 9000 9001 --name bastion --full true
    ```


