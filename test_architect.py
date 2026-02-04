from architect import GenericArchitect

def test_mock_architect():
    print("Testing Mock Architect...")
    arch = GenericArchitect(api_key="fake", mock_mode=True)

    cdl, error = arch.generate_cdl("Limit users to 5", "passport...")

    if error:
        print(f"Error: {error}")
    else:
        print("Success! CDL Generated:")
        print(cdl)
        assert cdl["description"] == "Mock Constraint: Limit users to 5"
        assert cdl["params"]["scope"]["type"] == "GLOBAL"

if __name__ == "__main__":
    test_mock_architect()
