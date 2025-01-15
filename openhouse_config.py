import os
import trino

connection_params = {
    'host': os.getenv('TRINO_HOST', '******.corp.******.com'),
    'port': int(os.getenv('TRINO_PORT', '***')),
    'user': os.getenv('TRINO_USER'),
    'catalog': os.getenv('TRINO_CATALOG', 'openhouse'),
    'http_scheme': os.getenv('TRINO_SCHEME', 'https'),
    'auth': trino.auth.BasicAuthentication(
        os.getenv('TRINO_USER'),
        os.getenv('TRINO_PASSWORD')
    )
} 