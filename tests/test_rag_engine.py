from part1_rag.rag_engine import DocumentStore, GroundedAssistant


def test_empty_document_directory_returns_no_retrieval_results(tmp_path):
    store = DocumentStore(tmp_path)

    assert store.retrieve("anything") == []
    result = GroundedAssistant(tmp_path).answer("Who is the supplier?")
    assert result["sources"] == []
    assert "would not invent" in result["answer"]


def test_transfer_fee_conflict_returns_both_source_documents():
    result = GroundedAssistant("data").answer("What is the transfer fee?")

    assert "conflict" in result["answer"].lower()
    assert {source["document"] for source in result["sources"]} == {
        "02_price_list_payment_plan.md",
        "03_booking_policy_faq.md",
    }


def test_missing_rental_yield_does_not_invent_a_number():
    result = GroundedAssistant("data").answer("What is the rental yield?")

    assert "do not provide a rental-yield figure" in result["answer"]
    assert result["sources"][0]["document"] == "03_booking_policy_faq.md"


def test_unconfirmed_tenant_is_reported_as_unconfirmed():
    result = GroundedAssistant("data").answer("Who is the anchor tenant?")

    assert "No anchor tenant is confirmed" in result["answer"]
    assert result["sources"][0]["document"] == "01_brochure.md"


def test_price_answer_has_a_supporting_source():
    result = GroundedAssistant("data").answer("What is the base price of a 2-bed in Block B?")

    assert "PKR 22,425,000" in result["answer"]
    assert result["sources"][0]["document"] == "02_price_list_payment_plan.md"

