package com.app.madcampweek3;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.ArrayList;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

public class NoteAdapter extends RecyclerView.Adapter<NoteAdapter.NoteViewHolder> {
    private ArrayList<NoteItem> noteList;

    public NoteAdapter(ArrayList<NoteItem> nList){
        this.noteList = nList;
    }

    // 레이아웃 파일에 있는 UI 컴포넌트를 NoteViewHolder 클래스의 멤버 변수와 연결
    public class NoteViewHolder extends RecyclerView.ViewHolder{
        TextView noteName;
        ImageView noteImg;

        public NoteViewHolder(View view){
            super(view);
            noteName = (TextView)view.findViewById(R.id.noteName);
            noteImg = (ImageView)view.findViewById(R.id.noteImg);
        }
    }

    @NonNull
    @Override
    public NoteViewHolder onCreateViewHolder(@NonNull ViewGroup viewGroup, int viewType) {
        View view = LayoutInflater.from(viewGroup.getContext()).inflate(R.layout.note_item,viewGroup,false);
        return new NoteAdapter.NoteViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull NoteViewHolder viewHolder, int position) {
        viewHolder.noteName.setText(noteList.get(position).getMock_name());
        if(noteList.get(position).getMock_subject().equals("수학")){
            viewHolder.noteImg.setImageResource(R.drawable.ic_math);
        }else if(noteList.get(position).getMock_subject().equals("국어")){
            viewHolder.noteImg.setImageResource(R.drawable.title);
        }else{
            viewHolder.noteImg.setImageResource(R.drawable.title);
        }
    }

    @Override
    public int getItemCount() {
        return (null != noteList ? noteList.size() : 0);
    }
}
