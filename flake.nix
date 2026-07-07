{
  description = "extraction-gym dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      fenix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        # toolchain = fenix.packages.${system}.stable.toolchain;

        # toolchain = fenix.packages.${system}.latest.toolchain;

        # toolchain = fenix.packages.${system}.fromToolchainFile {
        #   file = ./rust-toolchain.toml;
        #   sha256 = pkgs.lib.fakeSha256; # replace with real hash after first build
        # };

        # # Stable with custom components, commented out since we pinned to 1.87
        # toolchain = fenix.packages.${system}.stable.withComponents [
        #   "cargo"
        #   "rustc"
        #   "rust-src"
        #   "rustfmt"
        #   "clippy"
        #   "rust-analyzer"
        # ];

        # Pinned to the channel in ./rust-toolchain.toml
        toolchain =
          (fenix.packages.${system}.toolchainOf {
            channel = "1.87.0";
            sha256 = "sha256-KUm16pHj+cRedf8vxs/Hd2YWxpOrWZ7UOrwhILdSJBU=";
          }).withComponents
            [
              "cargo"
              "rustc"
              "rust-src"
              "rustfmt"
              "clippy"
              "rust-analyzer"
            ];
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = [
            toolchain
          ]
          ++ (with pkgs; [
            gnumake
            lldb
            clang
            lld
            pkg-config
            uv
            cargo-nextest
            cargo-expand
            cargo-show-asm
            cargo-unused-features
            cargo-wizard
            cargo-udeps
            cargo-sweep
            samply
            nil
            nixd
            nixfmt
          ]);

          buildInputs = with pkgs; [ cbc ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [ cbc ]);
        };
      }
    );
}
