"""Evidence falsification must preserve numeric meaning and selected rows."""
import unittest

from constraint_memory.memory.grounding import numeric_literal_present, property_row_proof


class GroundingTests(unittest.TestCase):
    def test_numeric_equivalence_and_signs(self):
        for value, source, expected in (
            (80, "80.00 USD", True), (80.0, "8e1 USD", True),
            (20, "-20 USD", False), (-20, "20 USD", False),
            (-20, "\u221220 USD", True), (20, "+20 USD", True),
            (40, "400 USD", False), (40, "40.01 USD", False),
            (40, "40,000 USD", False), (40000, "40,000 USD", True),
            (40, "40, please", True), (1, "1e-3 USD", False),
        ):
            with self.subTest(value=value, source=source):
                self.assertEqual(numeric_literal_present(value, source), expected)

    def test_dataframe_rows_bind_each_attribute_to_selected_entity(self):
        table = (
            "          NAME  price    room type house_rules  minimum nights  maximum occupancy  review rate number   city\n"
            "       Hotel A   80.0 Private room  No smoking             1.0                2.0                 4.5 Boston\n"
            "Hotel B longer   20.0  Shared room     No pets             2.0                4.0                 3.9 Boston"
        )
        for attribute, value, valid in (("price", 80, True), ("price", 20, False),
                                        ("rating", 4.5, True), ("rating", 3.9, False),
                                        ("room_type", "Private room", True)):
            with self.subTest(attribute=attribute, value=value):
                proven, _ = property_row_proof("Hotel A, Boston", "accommodation", attribute, value, [table])
                self.assertEqual(proven, valid)

    def test_city_and_duplicate_row_ambiguity_fail_closed(self):
        rows = [{"NAME": "Hotel A", "city": "Boston", "price": 80},
                {"NAME": "Hotel A", "city": "Paris", "price": 20}]
        self.assertTrue(property_row_proof("Hotel A, Boston", "accommodation", "price", 80, [rows])[0])
        self.assertFalse(property_row_proof("Hotel A, Boston", "accommodation", "price", 20, [rows])[0])
        rows.append({"NAME": "Hotel A", "city": "Boston", "price": 90})
        self.assertEqual(property_row_proof("Hotel A, Boston", "accommodation", "price", 80, [rows]),
                         (False, "PROPERTY_AMBIGUOUS_ENTITY"))

    def test_unknown_format_cannot_certify_a_numeric_property(self):
        proven, code = property_row_proof("Hotel A, Boston", "accommodation", "price", 20,
                                          ["Hotel A and Hotel B. Some price somewhere is 20."])
        self.assertFalse(proven)
        self.assertEqual(code, "PROPERTY_UNVERIFIABLE_ROW")


if __name__ == "__main__":
    unittest.main()
