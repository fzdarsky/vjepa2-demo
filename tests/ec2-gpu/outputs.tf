output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.gpu_test.id
}

output "public_ip" {
  description = "Public IP of the GPU test instance"
  value       = aws_instance.gpu_test.public_ip
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh ec2-user@${aws_instance.gpu_test.public_ip}"
}

output "cloud_init_log" {
  description = "Command to check cloud-init progress"
  value       = "ssh ec2-user@${aws_instance.gpu_test.public_ip} 'sudo tail -f /var/log/cloud-init-output.log'"
}
