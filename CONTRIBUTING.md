# Contributing to OTX

We welcome your input! 👐

We want to make it as simple and straightforward as possible to contribute to this project, whether it is a:

- Bug Report
- Discussion
- Feature Request
- Creating a Pull Request (PR)
- Becoming a maintainer

## Bug Report

We use GitHub issues to track the bugs. Report a bug by using our Bug Report Template in [Issues](https://github.com/open-edge-platform/training_extensions/issues/new?template=bug_report.md).

## Discussion

We enabled [GitHub Discussions](https://github.com/open-edge-platform/training_extensions/discussions) in OTX to welcome the community to ask questions and/or propose ideas/solutions. This will not only provide a medium for the community to discuss OTX but also help us de-clutter [Issues](https://github.com/open-edge-platform/training_extensions/issues/).

## Feature Request

We utilize GitHub issues to track the feature requests as well. If you are certain regarding the feature you are interested and have a solid proposal, you could then create the feature request by using our [Feature Request Template](https://github.com/open-edge-platform/training_extensions/issues/new?template=feature_request.md) in Issues. If it's still in the idea phase, you could discuss it with the community in our [Discussion](https://github.com/open-edge-platform/training_extensions/discussions/categories/ideas).

## Development & PRs

We actively welcome your pull requests:

### Getting Started

#### 1. Fork and Clone the Repository

First, fork the OTX repository by following the GitHub documentation on [forking a repo](https://docs.github.com/en/enterprise-cloud@latest/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo). Then, clone your forked repository to your local machine and create a new branch from `develop`.

#### 2. Set Up Your Development Environment

Set up your development environment to start contributing. This involves installing the required dependencies and setting up pre-commit hooks for code quality checks. Note that this guide assumes you are using [Venv](https://docs.python.org/3/library/venv.html) for python environments management. However, the steps are similar for other env managers.

<details>
<summary>Development Environment Setup Instructions</summary>

1. Create and activate a new python environment:

   ```bash
   cd lib
   python -m venv .otx
   source .otx/bin/activate
   ```

2. Install the development requirements:

   ```bash
   pip install -e .[dev]
   ```

Make sure to address any pre-commit issues before finalizing your pull request.
Pre-commit checks can be launched by the command:

```bash
tox -vv -e pre-commit
```

</details>

### Making Changes

1. **Write Code:** Follow the project's coding standards and write your code with clear intent. Ensure your code is well-documented and includes examples where appropriate. For code quality we use ruff, whose configuration is in [`pyproject.toml`](pyproject.toml) file.

2. **Add Tests:** If your code includes new functionality, add corresponding tests using [pytest](https://docs.pytest.org/en/7.4.x/) to maintain coverage and reliability.

3. **Update Documentation:** If you've changed APIs or added new features, update the documentation accordingly. Ensure your docstrings are clear and follow [Google's docstring guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

4. **Pass Tests and Quality Checks:** Ensure the test suite passes and that your code meets quality standards by running:

   ```bash
   cd lib
   tox -vv -e pre-commit
   pytest tests/unit
   pytest tests/integration
   ```

5. **Update the Changelog:** For significant changes, add a summary to the [CHANGELOG](CHANGELOG.md).

6. **Check Licensing:** Ensure you own the code or have rights to use it, adhering to appropriate licensing.

7. **Sign Your Commits:** Use signed commits to certify that you have the right to submit the code under the project's license:

   ```bash
   git commit -S -m "Your detailed commit message"
   ```

   For more on signing commits, see [GitHub's guide on signing commits](https://docs.github.com/en/github/authenticating-to-github/managing-commit-signature-verification/signing-commits).

### Submitting Pull Requests

Once you've followed the above steps and are satisfied with your changes:

1. Push your changes to your forked repository.
2. Go to the original OTX repository you forked and click "New pull request".
3. Choose your fork and the branch with your changes to open a pull request.
4. Fill in the pull request template with the necessary details about your changes.

We look forward to your contributions!

## License

You accept that your contributions will be licensed under the [Apache-2.0 License](https://choosealicense.com/licenses/apache-2.0/) if you contribute to this repository. If this is a concern, please notify the maintainers.
