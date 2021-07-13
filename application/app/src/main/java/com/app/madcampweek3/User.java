package com.app.madcampweek3;

public class User {
    public static String email = "osjosj1@gmail.com";

    public User(String email) {
        this.email = email;
    }

    public static String getEmail() {
        return email;
    }

    public static void setEmail(String email) {
        User.email = email;
    }
}
