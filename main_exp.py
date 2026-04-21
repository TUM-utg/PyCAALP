from pycaalp.run import create_assembly_digraph, optimize


if __name__ == "__main__":
    PART_FNAME = "data/assembly_1/assembly_1_parts.json"

    caalp = create_assembly_digraph(
        file_name=PART_FNAME, w_tech=0.0, w_hand=0.0, w_tol=0.0, w_mass=1.0
    )
    print(f"{caalp.one_assembly_policy=}")
    result = optimize(
        assembly_digraph=caalp, return_model=True, num_phases=3, w_balanced=0.5
    )
    print(result)
