package com.app.madcampweek3;

import android.app.LauncherActivity;
import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.ImageView;

import java.util.ArrayList;

public class CaptureAdapter extends BaseAdapter {
    ArrayList<CaptureItem> items = new ArrayList<CaptureItem>();
    Context context; // 어플리케이션에 대한 정보를 담고 있는 context 객체

    //  ArrayList에 ListItem 객체를 추가하기 위한 메서드
    public void addItem(CaptureItem item){
        items.add(item);
    }

    @Override
    public int getCount() {
        return items.size();
    }

    @Override
    public CaptureItem getItem(int position) {
        return items.get(position);
    }

    @Override
    public long getItemId(int position) {
        return position;
    }

    @Override
    public View getView(int position, View view, ViewGroup viewGroup) {
        context = viewGroup.getContext(); // 아이템의 부모 레이아웃을 설정
        CaptureItem captureItem = items.get(position); // position에 해당하는 listItem

        // 아이템 레이아웃(capture_item.xml)을 위한 파일을 inflate하고 view를 참조
        if(view == null){
            LayoutInflater inflater = (LayoutInflater)context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
            view = inflater.inflate(R.layout.capture_item, viewGroup, false);
        }

        ImageView imageView = view.findViewById(R.id.capture);
        imageView.setImageURI(captureItem.getImgUri());

        return view;
    }
}
