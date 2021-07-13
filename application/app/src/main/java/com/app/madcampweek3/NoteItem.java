package com.app.madcampweek3;

public class NoteItem {
    private String mock_id;
    private String mock_name;
    private String mock_subject;

    public NoteItem(String mock_id, String mock_name, String mock_subject) {
        this.mock_id = mock_id;
        this.mock_name = mock_name;
        this.mock_subject = mock_subject;
    }

    public String getMock_id() {
        return mock_id;
    }

    public void setMock_id(String mock_id) {
        this.mock_id = mock_id;
    }

    public String getMock_name() {
        return mock_name;
    }

    public void setMock_name(String mock_name) {
        this.mock_name = mock_name;
    }

    public String getMock_subject() {
        return mock_subject;
    }

    public void setMock_subject(String mock_subject) {
        this.mock_subject = mock_subject;
    }
}
