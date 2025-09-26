import argparse
import sys


def main():
    """
    Main function for the phaze CLI.
    """
    parser = argparse.ArgumentParser(
        description="PHAZE: Cryptographic and ZKML-based low latency inference at LHC."
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=True,
        help="Running mode. Options: [new_project, ... (to be implemented)]",
    )

    parser.add_argument(
        "-p",
        "--project",
        nargs=2,
        metavar=("WORKSPACE_NAME", "PROJECT_NAME"),
        help="Workspace and project name.",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output."
    )

    args = parser.parse_args()

    if args.verbose:
        print("Verbose mode enabled.")
        print(f"Running in mode: {args.mode}")
        if args.project:
            print(f"Workspace: {args.project[0]}")
            print(f"Project: {args.project[1]}")

    # Placeholder for mode handling
    if args.mode == "new_project":
        if not args.project:
            print(
                "Error: --project argument is required for 'new_project' mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            f"Creating new project '{args.project[1]}' in workspace '{args.project[0]}'... (Not yet implemented)"
        )
    else:
        print(f"Mode '{args.mode}' is not yet implemented.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
