package com.app.madcampweek3;

public class QandAItem {
    private String qanda;
    private String q_image;
    private String explanation;

    public QandAItem(String qanda, String q_image, String explanation) {
        this.qanda = qanda;
        this.q_image = q_image;
        this.explanation = explanation;
    }

    public String getQanda() {
        return qanda;
    }

    public void setQanda(String qanda) {
        this.qanda = qanda;
    }

    public String getQ_image() {
        return q_image;
    }

    public void setQ_image(String q_image) {
        this.q_image = q_image;
    }

    public String getExplanation() {
        return explanation;
    }

    public void setExplanation(String explanation) {
        this.explanation = explanation;
    }
}
