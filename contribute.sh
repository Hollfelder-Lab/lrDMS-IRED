echo "Setting up the project for development..."
cp .env.sample .env
pip install -e ".[dev]"
echo "You're all set! Thanks for contributing to the project."