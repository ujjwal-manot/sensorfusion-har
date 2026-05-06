import subprocess
import sys
from pathlib import Path


def generate_cert(cert_dir: Path, hostname: str = "localhost"):
    """Generate self-signed SSL certificate for HTTPS."""
    cert_dir = Path(cert_dir)
    cert_dir.mkdir(parents=True, exist_ok=True)
    
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"
    
    # Check if cert already exists
    if cert_file.exists() and key_file.exists():
        return cert_file, key_file
    
    # Generate self-signed certificate
    try:
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key_file),
            "-out", str(cert_file),
            "-days", "365",
            "-nodes",
            "-subj", f"/CN={hostname}"
        ], check=True, capture_output=True)
        
        print(f"Generated SSL certificate: {cert_file}")
        print(f"Generated SSL key: {key_file}")
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate certificate: {e}")
        print("Falling back to certifi if available")
        
        # Fallback: use certifi
        try:
            import certifi
            import ssl
            cert_file = Path(certifi.where())
            key_file = cert_file  # Same file for certifi
            print(f"Using certifi certificate: {cert_file}")
        except ImportError:
            print("certifi not installed, SSL may not work properly")
    
    return cert_file, key_file


if __name__ == "__main__":
    cert_dir = Path("certs")
    hostname = "localhost"
    
    if len(sys.argv) > 1:
        hostname = sys.argv[1]
    
    cert_file, key_file = generate_cert(cert_dir, hostname)
    print(f"\nCertificate: {cert_file}")
    print(f"Key: {key_file}")
