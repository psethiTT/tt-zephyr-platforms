import pyluwen

def test_set_characterisation_host_fmin():
    """
    Validates that the SET_HOST_REQUESTED_FMIN characterization message works.

    Sets a host fmin of 600 MHz, which should be within the valid range
    [AICLK_FMIN_MIN=200, AICLK_FMIN_MAX=1400] on all boards.
    Verifies that the request succeeds.
    """
    arc_chip = pyluwen.detect_chips()[0]

    NEW_HOST_FMIN = 900  # MHz

    # Positive case: set a valid host fmin
    response = arc_chip.as_bh().arc_msg_buf(
        [
            0x1C6,
            NEW_HOST_FMIN,
            0,
            0,
            0,
            0,
            0,
            0
        ]
    )
    assert response[0] == 0, f"Failed to set host fmin to {NEW_HOST_FMIN} MHz"
    print(f"Successfully set host fmin to {NEW_HOST_FMIN} MHz")
    
test_set_characterisation_host_fmin()