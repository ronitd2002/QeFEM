def show_equilibrium_chain(
    N,
    nu=100e3,
    M=40 * atomic_mass
):
    """
    Find, print, and plot the equilibrium positions of N ions.

    Parameters
    ----------
    N : int
        Number of ions.

    nu : float
        Axial trap frequency nu/(2*pi), in Hz.
        Default: 100 kHz.

    M : float
        Mass of one ion, in kg.
        Default: mass of 40Ca+.

    Returns
    -------
    u : numpy array
        Dimensionless equilibrium positions.

    x_m : numpy array
        Dimensional equilibrium positions in metres.
    """

    # Paper's nu is an angular frequency
    nu_angular = 2 * np.pi * nu

    # Calcium II = Ca+, so Z = 1
    Z = 1

    # Equation (2.4)
    ell_local = (
        Z**2 * elementary_charge**2
        / (4 * np.pi * epsilon_0 * M * nu_angular**2)
    )**(1 / 3)

    # Solve Equation (2.5)
    u = solve_equilibrium(N)

    # Dimensional coordinates
    x_m = ell_local * u
    x_um = x_m * 1e6

    # Print results
    print(f"N = {N}")
    print(f"Trap frequency = {nu / 1e3:.3f} kHz")
    print(f"Ion mass = {M:.6e} kg")
    print(f"Length scale ell = {ell_local * 1e6:.6f} micrometres")

    print("\nDimensionless positions u_m:")
    print(
        np.array2string(
            u,
            precision=6,
            separator=", ",
            max_line_width=120
        )
    )

    print("\nPhysical positions x_m^(0) [micrometres]:")
    print(
        np.array2string(
            x_um,
            precision=6,
            separator=", ",
            max_line_width=120
        )
    )

    # Plot
    figure_width = min(max(8, 0.6 * N), 16)
    fig, ax = plt.subplots(figsize=(figure_width, 2.4))

    ax.axhline(0, color="grey", linewidth=1, alpha=0.5)

    ax.scatter(
        x_um,
        np.zeros(N),
        s=110,
        color="royalblue",
        edgecolor="black",
        zorder=3
    )

    # Avoid overcrowding for large chains
    if N <= 20:
        for ion_number, position in enumerate(x_um, start=1):
            ax.annotate(
                str(ion_number),
                (position, 0),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center"
            )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.35
    )

    ax.set_xlabel(r"Equilibrium position $x_m^{(0)}$ [$\mu$m]")
    ax.set_title(
        rf"{N} trapped $^{{40}}\mathrm{{Ca}}^+$ ions, "
        rf"$\nu/(2\pi)={nu / 1e3:g}$ kHz"
    )

    ax.set_yticks([])
    ax.set_ylim(-0.15, 0.15)
    ax.grid(axis="x", alpha=0.25)
    ax.margins(x=0.08)

    plt.tight_layout()
    plt.show()

    return u, x_m